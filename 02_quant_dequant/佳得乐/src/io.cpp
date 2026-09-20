#include "project.hpp"
#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
namespace lp {
static std::string trim(std::string s) {
    //" \t\r\n\""  也就是：空格、Tab、\r、\n、双引号 "。
    //找到左边第一个“不是这些字符”的位置。
    auto b = s.find_first_not_of(" \t\r\n\"");
    //如果整个字符串全都是这些字符，就返回空串。
    if (b == std::string::npos)
        return "";
    //再找到右边最后一个“不是这些字符”的位置，然后截取中间部分。
    return s.substr(b, s.find_last_not_of(" \t\r\n\"") - b + 1);
}

//把字符串映射为它在给定列表中的下标（从 0 开始）
static int choice(const std::string &s, std::initializer_list<const char *> values) {
    int i = 0;
    for (auto v : values) {
        if (s == v)
            return i;
        ++i;
    }
    throw std::runtime_error("invalid value: " + s);
}

//配置文件解析函数：读取 key=value 形式的文本，校验键名，去重，然后根据解析出的 map 构造 Config 对象。
Config read_config(const std::string &path) {
    //创建输入文件流 in，尝试打开 path 指定的文件。
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open config: " + path);
    std::map<std::string, std::string> m;
    std::string line;
    std::set<std::string> keys = {"format",      "fp8_type", "block_size", "scale_mode",
                                  "output_type", "rounding", "target_gpu", "seed"};
    //循环从文件流 in 中读取一行到 line，直到文件结束或读取失败。
    while (std::getline(in, line)) {
        //查找当前行中第一个 #，截取它之前的部分作为有效内容。如果找不到 #，find 返回 std::string::npos，substr(0, npos) 会取整个字符串。
        //这一步实现“去掉注释”。
        line = line.substr(0, line.find('#'));
        //调用之前定义的 trim 去掉两端空白和双引号，如果结果为空，说明这一行是空行或只有注释/空白。
        if (trim(line).empty())
            //跳过这一行，继续读下一行。
            continue;
        //在当前行中查找等号 = 的位置。
        auto p = line.find('=');
        if (p == std::string::npos)
            throw std::runtime_error("config requires key=value");
        //k：等号左边的子串，trim 后作为键。
        //v：等号右边的子串，trim 后作为值。
        auto k = trim(line.substr(0, p)), v = trim(line.substr(p + 1));
        //!keys.count(k)：如果 k 不在允许的键集合中。
        //!m.emplace(k, v).second：尝试把 k,v 插入 map，如果插入失败（second 为 false），说明键 k 已经存在。
        if (!keys.count(k) || !m.emplace(k, v).second)
            throw std::runtime_error("unknown/duplicate key: " + k);
    }
    Config c;
    //如果配置里存在 format 键。
    if (m.count("format"))
        // 调用 choice 把 format 的值映射成索引：
        // "mxfp8" → 0
        // "nvfp4" → 1
        // 然后三元运算符：如果 choice 返回非零（即 1，对应 nvfp4），则 c.format = 2；否则 c.format = 0。所以最终：
        // mxfp8 → c.format = 0
        // nvfp4 → c.format = 2
        c.format = choice(m["format"], {"mxfp8", "nvfp4"}) ? 2 : 0;
    //如果配置里存在 fp8_type 键。
    if (m.count("fp8_type")) {
        //如果前面已经把 c.format 设成了 2，也就是 nvfp4。
        if (c.format == 2)
            //抛出异常，因为 nvfp4 不使用 fp8_type。
            throw std::runtime_error("fp8_type not applicable to nvfp4");
        c.format = choice(m["fp8_type"], {"e4m3", "e5m2"});
    }
    //根据 c.format 设置默认块大小：如果是 nvfp4（c.format == 2），默认 block = 16。
    //                           否则默认 block = 32。
    c.block = c.format == 2 ? 16 : 32;

    //定义一个 lambda 表达式 integer，接受一个 const std::string&，用于把字符串解析成非负整数。
    auto integer = [](const std::string &s) {
        //s.empty()：空串不合法。
        //s.find_first_not_of("0123456789") != std::string::npos：找到任何一个不是数字的字符，说明含有非数字，不合法。
        if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("invalid unsigned integer");
        size_t end;
        //把字符串 s 转换成 unsigned long long，end 会记录解析到第几个字符。
        auto v = std::stoull(s, &end);
        if (end != s.size())
            throw std::runtime_error("invalid integer");
        return v;
    };

    //如果配置里有 block_size。
    if (m.count("block_size")) {
        auto v = integer(m["block_size"]);
        if (v > INT32_MAX)
            throw std::runtime_error("block too large");
        c.block = int(v);
    }
    if (m.count("seed"))
        c.seed = integer(m["seed"]);
    if (m.count("scale_mode"))
        c.mode = choice(m["scale_mode"], {"block", "tensor"});
    if (m.count("output_type"))
        c.out = choice(m["output_type"], {"fp32", "fp16", "bf16"});
    if (m.count("rounding"))
        c.round = choice(m["rounding"], {"nearest", "stochastic"});
    if (m.count("target_gpu"))
        c.target = m["target_gpu"];
    validate(c);
    return c;
}


using Bytes = std::vector<uint8_t>;
// 定义一个 static 函数 put：b：目标字节缓冲区，按引用传入，函数会往它末尾追加内容。
//                         x：要写入的整数值，类型 uint64_t，最多 8 字节。
//                         n：写几个字节，调用方传 1、4、8 等。
// static 表示内部链接，只在本 .cpp 文件可见。

// 以 put(b, 0x01020304, 4) 为例：
// i	x >> 8*i	uint8_t结果	    追加的字节
// 0	0x01020304	    0x04	      04
// 1	0x00010203	    0x03	      03
// 2	0x00000102	    0x02	      02
// 3	0x00000001	    0x01	      01

// 最终 b 末尾追加的是 04 03 02 01，低字节在前，即小端序（little-endian）。
static void put(Bytes &b, uint64_t x, int n) {
    for (int i = 0; i < n; ++i)
        //uint8_t(...) 只保留最低 8 位
        b.push_back(uint8_t(x >> (8 * i)));
}

//定义一个结构体 Reader，用来封装“从字节缓冲区读取”的状态和操作。
struct Reader {
    //成员变量 b，类型是之前定义的 Bytes（即 std::vector<uint8_t>）。它持有整个文件内容。注意这里是按值存一份拷贝，不是引用。
    Bytes b;
    //成员变量 pos，记录当前读到第几个字节。默认从 0 开始。
    size_t pos = 0;
    // 例如 put(b, 0x01020304, 4) 写入 04 03 02 01，get(4) 读回：
    // i	读到的字节	    左移	        或入 x
    // 0	0x04	      0x04	          0x04
    // 1	0x03	      0x0300	      0x0304
    // 2	0x02	      0x020000	      0x020304
    // 3	0x01	      0x01000000	  0x01020304
    uint64_t get(int n) {
        if (pos > b.size() || size_t(n) > b.size() - pos)
            throw std::runtime_error("truncated file");
        uint64_t x = 0;
        for (int i = 0; i < n; ++i)
            x |= uint64_t(b[pos++]) << (8 * i);
        return x;
    }
};

//这是运行时字节序检查函数，用来确保当前主机是小端序（little-endian），否则直接报错。
static void endian_check() {
    uint32_t tag = 1;
    // tag 是 uint32_t，是一个内存对象。
    // &tag：取它在内存中的地址。
    // reinterpret_cast<uint8_t*>(&tag)：把地址重新解释为“指向 1 字节”的指针。
    // *...：读那个地址处的 1 个字节。
    //读的是 tag 占用内存的第一个字节
    if (*reinterpret_cast<uint8_t *>(&tag) != 1)
        throw std::runtime_error("big endian host not supported");
}

static Bytes read(const std::string &path) {
    //先检查主机是不是小端序。不是就抛异常。因为后续序列化代码依赖小端，所以读文件前先过这道关。
    endian_check();
    //以二进制模式打开文件，并且用 std::ios::ate（at end）把读写位置初始定位到文件末尾。这样打开后就能立刻用 tellg() 拿到文件大小。
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error("cannot open: " + path);
    //tellg() 返回当前读取位置，因为用了 ate，这里就是文件末尾位置，也就是文件大小。类型是 std::streampos。
    auto len = in.tellg();
    if (len < 0)
        throw std::runtime_error("file size");
    //构造一个大小为 len 的 Bytes，也就是分配好能装下整个文件的缓冲区。static_cast<size_t> 把 streampos 转成无符号大小类型。
    Bytes b(static_cast<size_t>(len));
    //把读取位置移回文件开头。因为之前用 ate 定位在末尾，不回到开头的话，下面的 read 会什么都读不到。
    in.seekg(0);
    //一次性把 b.size() 个字节读进缓冲区：
    //b.data()：返回 uint8_t*，指向缓冲区首地址。
    //reinterpret_cast<char *>(...)：std::istream::read 要求 char*，所以把 uint8_t* 重新解释为 char*。这只是类型转换，不改数据。
    //in.read(...) 返回流本身，!in 在读取失败（比如文件比预期短）时为真。
    //失败就抛 "file read failed"。
    if (!in.read(reinterpret_cast<char *>(b.data()), b.size()))
        throw std::runtime_error("file read failed");
    return b;
}

static void write(const std::string &path, const Bytes &b) {
    endian_check();
    // 以二进制模式打开输出文件流：
    // std::ios::binary：不做文本转换，避免 Windows 上 \n 被写成 \r\n。
    // 如果文件不存在会创建；存在会截断（默认 std::ios::trunc）。
    std::ofstream out(path, std::ios::binary);
    // 1.!out：判断文件是否打开成功。失败就直接短路，不执行后面的 write。
    // 2.!out.write(...)：如果打开成功，调用 write 写入数据。
    //   b.data()：返回 const uint8_t*，指向缓冲区首地址。
    //   reinterpret_cast<const char *>(...)：std::ostream::write 要求 const char*，所以把 const uint8_t* 重新解释为 const char*。只改类型，不改数据。
    //   b.size()：要写的字节数。
    //   out.write 返回流本身，!out 在写入失败（磁盘满、权限不足等）时为真。
    if (!out || !out.write(reinterpret_cast<const char *>(b.data()), b.size()))
        throw std::runtime_error("file write failed: " + path);
    //显式关闭文件流。关闭会刷新缓冲区，把剩余数据真正写到磁盘。
    out.close();
    if (!out)
        throw std::runtime_error("file close failed");
}

//写入端
static void prefix(Bytes &b, const char *magic) {
    //把 magic 的前 8 个字符逐个追加到字节缓冲区 b。注意这里用的是 char，直接转成 uint8_t 存进去（ASCII 码）。
    for (int i = 0; i < 8; ++i)
        b.push_back(magic[i]);
    //写入版本号 1，4 字节小端：01 00 00 00。
    put(b, 1, 4);
    //写入端序标签 0x01020304，4 字节小端：04 03 02 01。
    put(b, 0x01020304, 4);
}

//读取端
static void prefix(Reader &r, const char *magic) {
    //从流里读 8 个字节，逐个和期望的 magic 比较。任何一个不符，就抛 "invalid magic"，说明这不是对应类型的文件（或者文件损坏、格式不对）。
    for (int i = 0; i < 8; ++i)
        if (r.get(1) != uint8_t(magic[i]))
            throw std::runtime_error("invalid magic");
    //读 4 字节版本号，必须等于 1。不等于就抛 "unsupported version"，说明文件是别的版本写的，当前代码不认识。
    if (r.get(4) != 1)
        throw std::runtime_error("unsupported version");
    //读 4 字节端序标签，必须等于 0x01020304。不等于就抛 "invalid endian tag"。
    if (r.get(4) != 0x01020304)
        throw std::runtime_error("invalid endian tag");
}


void save_tensor(const std::string &path, const Tensor &t) {
    auto n = checked_size(t.rows, t.cols);
    // 校验张量是否自洽，三个条件：1. t.dtype < 0 || t.dtype > 2：dtype 只允许 0、1、2。
    //                        2. t.data.size() != n * (t.dtype ? 2 : 4)：数据字节数必须和元素数匹配。
    //                             dtype == 0：每个元素 4 字节（推测是 fp32）。
    //                             dtype == 1 或 2：每个元素 2 字节（推测是 fp16、bf16）。
    //                         任何一个不满足就抛 "invalid tensor"。
    if (t.dtype < 0 || t.dtype > 2 || t.data.size() != n * (t.dtype ? 2 : 4))
        throw std::runtime_error("invalid tensor");
    Bytes b;
    //写入 16 字节文件头：8 字节魔数 "LPTENSOR"、4 字节版本号 1、4 字节端序标签 0x01020304。
    prefix(b, "LPTENSOR");
    //写入行数，8 字节小端。
    put(b, t.rows, 8);
    //写入列数，8 字节小端。
    put(b, t.cols, 8);
    //写入 dtype，4 字节小端。
    put(b, t.dtype, 4);
    //写入一个固定值 48，4 字节小端。这是头部总大小的标记，用于加载时校验。
    put(b, 48, 4);
    //写入数据字节数，8 字节小端。加载时用来校验数据长度。
    put(b, t.data.size(), 8);
    //把张量的原始数据追加到缓冲区末尾。t.data 是 std::vector<uint8_t>，直接整块拷贝。
    b.insert(b.end(), t.data.begin(), t.data.end());
    //调用 write 把整个 b 写到 path。
    write(path, b);
}


Tensor load_tensor(const std::string &path) {
    //read(path)：把整个文件读成 Bytes。
    //Reader r{...}：用这个 Bytes 初始化 Reader，r.pos 从 0 开始。

    // Reader r{read(path)};等价于：Reader r = {read(path)};
    // 花括号里第一个值给第一个成员 b，第二个值给第二个成员 pos，以此类推。

    // 花括号里只有一个值 read(path)，它给第一个成员 b。
    // 第二个成员 pos 没有提供值，就使用它的默认成员初始化器 = 0。
    // 所以等价于：Reader r;
    //           r.b = read(path);   // 实际上是移动/拷贝初始化
    //           r.pos = 0;
    Reader r{read(path)};
    // 调用 prefix 的读取版（因为第一个参数是 Reader）。校验文件头 16 字节：
    // 前 8 字节必须是 "LPTENSOR"；
    // 接下来 4 字节版本号必须是 1；
    // 再 4 字节端序标签必须是 0x01020304。
    // 任何一项不符就抛异常。通过后 r.pos 前进到 16。 prefix()中的get()中每次都会pos++
    prefix(r, "LPTENSOR");
    Tensor t;
    t.rows = r.get(8);
    t.cols = r.get(8);
    auto n = checked_size(t.rows, t.cols);
    auto dt = r.get(4);
    if (dt > 2)
        throw std::runtime_error("invalid dtype");
    t.dtype = dt;
    if (r.get(4) != 48 || r.get(8) != n * (t.dtype ? 2 : 4) ||
        r.b.size() - r.pos != n * (t.dtype ? 2 : 4))
        throw std::runtime_error("invalid tensor payload");
    //把数据区拷进 t.data。此时 r.pos == 48，r.b.begin() + 48 指向数据区起点，r.b.end() 是文件末尾。
    //因为前面已经校验剩余长度等于 n * 每元素字节数，所以这里拷贝的字节数正好。
    t.data.assign(r.b.begin() + r.pos, r.b.end());
    return t;
}

// 返回一个位标志，用两个 bit 编码信息：
//     bit 0（值 1）：p.c.mode 是否为真。
//         mode == 1（tensor 模式）→ bit0 = 1。
//         mode == 0（block 模式）→ bit0 = 0。
//     bit 1（值 2）：是否“block 模式下每行最后一个块不满”。
//         沿行方向分块时，不完整块只可能出现在每行的末尾，
//         因此判断条件是 p.cols % p.c.block != 0，
//         而不是整个数组长度 p.size() 对块大小取模。
//         只有当 mode == 0（block 模式）且 p.cols % p.c.block != 0 时才置 1。
//         如果 mode == 1，这个 bit 永远是 0。
//
// extension 实际只可能是：
//     0：block 模式且每行都能整除（无行尾不满块）
//     2：block 模式且每行末尾有余（有行尾不满块）
//     1：tensor 模式
static unsigned extension(const Packed &p) {
    return (p.c.mode ? 1 : 0) | ((!p.c.mode && p.cols % p.c.block) ? 2 : 0);
}


void save_packed(const std::string &path, const Packed &p) {
    validate_packed(p);
    Bytes b;
    prefix(b, "LPPACKED");
    put(b, p.rows, 8);
    put(b, p.cols, 8);
    for (int v : {p.input_dtype, p.c.format, p.c.block, p.c.mode, p.c.out, p.c.round})
        put(b, v, 4);
    put(b, p.c.seed, 8);
    put(b, p.c.format == 2 ? 1 : 0, 4);
    //写入 extension 算出的标志，4 字节小端。加载端用它校验 mode 和部分块情况。偏移到 72。
    put(b, extension(p), 4);
    //把 global（float）的位模式转成 uint32_t，写 4 字节小端。
    put(b, bits(p.global), 4);

    //写入头部大小标记 112，4 字节小端。偏移到 80。
    //这个 112 是头部总大小，也就是 scales 数据开始之前的字节数。加载端会校验。
    put(b, 112, 4);
    //写入 scales 的字节数，8 字节小端。偏移到 88。
    put(b, p.scales.size(), 8);

    //写入 scales 的起始偏移 112，8 字节小端。偏移到 96。
    //因为头部固定 112 字节，scales 从 112 开始。
    put(b, 112, 8);
    //写入 data 的字节数，8 字节小端。偏移到 104。
    put(b, p.data.size(), 8);
    //写入 data 的起始偏移 112 + scales.size()，8 字节小端。偏移到 112。
    put(b, 112 + p.scales.size(), 8);
    //根据scales的偏移(put(b,112,8))，得到scales的数据从哪里开始
    b.insert(b.end(), p.scales.begin(), p.scales.end());
    //根据data的偏移，得到data的数据从哪里来
    b.insert(b.end(), p.data.begin(), p.data.end());
    write(path, b);
}


Packed load_packed(const std::string &path) {
    Reader r{read(path)};
    prefix(r, "LPPACKED");
    Packed p;
    p.rows = r.get(8);
    p.cols = r.get(8);
    checked_size(p.rows, p.cols);
    p.input_dtype = r.get(4);
    p.c.format = r.get(4);
    p.c.block = r.get(4);
    p.c.mode = r.get(4);
    p.c.out = r.get(4);
    p.c.round = r.get(4);
    p.c.seed = r.get(8);
    auto st = r.get(4), ext = r.get(4);
    p.global = frombits(r.get(4));
    auto header = r.get(4), ns = r.get(8), so = r.get(8), np = r.get(8), po = r.get(8);
    validate(p.c);
    uint64_t n = p.size();
    // 沿行方向分块后的块数
    uint64_t bpr = (p.cols + p.c.block - 1) / p.c.block;
    uint64_t expected_s = p.c.mode ? 1 : p.rows * bpr;
    uint64_t expected_p = p.c.format == 2 ? (n + 1) / 2 : n;
    if (header != 112 || st != unsigned(p.c.format == 2) || ext != extension(p) ||
        ns != expected_s || np != expected_p || so != 112 || po != 112 + ns || po > r.b.size() ||
        np != r.b.size() - po)
        throw std::runtime_error("invalid packed offsets/lengths/metadata");
    p.scales.assign(r.b.begin() + so, r.b.begin() + po);
    p.data.assign(r.b.begin() + po, r.b.end());
    validate_packed(p);
    return p;
}
} // namespace lp
