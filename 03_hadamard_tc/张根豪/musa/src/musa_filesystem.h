#pragma once

#include <cerrno>
#include <cstdio>
#include <fstream>
#include <string>
#include <sys/stat.h>

namespace fs {
class path {
 public:
  path() = default;
  path(const char* value) : value_(value) {}
  path(const std::string& value) : value_(value) {}
  operator std::string() const { return value_; }
  path operator/(const char* child) const {
    return path(value_.empty() || value_ == "." ? value_ + child
                                                 : value_ + "/" + child);
  }
  path parent_path() const {
    const std::string::size_type slash = value_.find_last_of('/');
    return slash == std::string::npos ? path() : path(value_.substr(0, slash));
  }
  bool empty() const { return value_.empty(); }
  const char* c_str() const { return value_.c_str(); }
  const std::string& string() const { return value_; }

 private:
  std::string value_;
};

inline bool exists(const path& value) {
  struct stat info;
  return ::stat(value.string().c_str(), &info) == 0;
}

inline std::size_t file_size(const path& value) {
  struct stat info;
  return ::stat(value.string().c_str(), &info) == 0
             ? static_cast<std::size_t>(info.st_size)
             : 0;
}

inline void create_directories(const path& value) {
  const std::string text = value.string();
  if (text.empty() || text == ".") return;
  std::string current;
  for (char character : text) {
    current += character;
    if (character == '/') ::mkdir(current.c_str(), 0755);
  }
  ::mkdir(current.c_str(), 0755);
}
}  // namespace fs