#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "file_io.h"
#include "search.h"

namespace {

float halfToFloat(std::uint16_t bits) {
  const bool negative = (bits & 0x8000U) != 0;
  const std::uint32_t exponent = (bits >> 10U) & 0x1fU;
  const std::uint32_t mantissa = bits & 0x03ffU;
  float value = 0.0F;
  if (exponent == 0) {
    value = std::ldexp(static_cast<float>(mantissa), -24);
  } else if (exponent == 31) {
    value = mantissa == 0 ? std::numeric_limits<float>::infinity()
                          : std::numeric_limits<float>::quiet_NaN();
  } else {
    value = std::ldexp(static_cast<float>(1024U + mantissa),
                       static_cast<int>(exponent) - 25);
  }
  return negative ? -value : value;
}

float databaseValue(const VectorDatabase& database, std::uint64_t index) {
  return database.dtype == DataType::kFloat32
             ? database.values[index]
             : halfToFloat(database.half_values[index]);
}

float queryValue(const QuerySet& queries, std::uint64_t index) {
  return queries.dtype == DataType::kFloat32
             ? queries.values[index]
             : halfToFloat(queries.half_values[index]);
}

float computeScore(const VectorDatabase& database, const QuerySet& queries,
                   std::uint64_t query_id, std::uint64_t vector_id) {
  float dot = 0.0F;
  float query_norm = 0.0F;
  float vector_norm = 0.0F;
  float squared_l2 = 0.0F;

  for (std::uint32_t d = 0; d < database.dim; ++d) {
    const float q = queryValue(queries, query_id * queries.dim + d);
    const float x = databaseValue(database, vector_id * database.dim + d);
    if (database.metric == Metric::kL2) {
      const float difference = q - x;
      squared_l2 = std::fma(difference, difference, squared_l2);
    } else {
      dot = std::fma(q, x, dot);
      if (database.metric == Metric::kCosine) {
        query_norm = std::fma(q, q, query_norm);
        vector_norm = std::fma(x, x, vector_norm);
      }
    }
  }

  if (database.metric == Metric::kL2) {
    return squared_l2;
  }
  if (database.metric == Metric::kInnerProduct) {
    return dot;
  }
  if (query_norm == 0.0F || vector_norm == 0.0F) {
    return 0.0F;
  }
  return dot / std::sqrt(query_norm * vector_norm);
}

bool isBetter(const Neighbor& left, const Neighbor& right, Metric metric) {
  if (left.score == right.score) {
    return left.id < right.id;
  }
  if (metric == Metric::kL2) {
    return left.score < right.score;
  }
  return left.score > right.score;
}

std::vector<Neighbor> selectTopK(std::vector<Neighbor> candidates,
                                 std::uint32_t top_k, Metric metric) {
  const auto comparator = [metric](const Neighbor& left,
                                   const Neighbor& right) {
    return isBetter(left, right, metric);
  };

  // nth_element 先把最好的 K 个移到前面，再只排序这 K 个。
  // 它比对全部候选排序更省时间，同时代码仍然很容易理解。
  if (top_k < candidates.size()) {
    std::nth_element(candidates.begin(), candidates.begin() + top_k,
                     candidates.end(), comparator);
    candidates.resize(top_k);
  }
  std::sort(candidates.begin(), candidates.end(), comparator);
  return candidates;
}

}  // namespace

SearchResults cpuExactSearch(const VectorDatabase& database,
                             const QuerySet& queries,
                             const SearchParams& params) {
  validateInputs(database, queries, params);
  SearchResults results(queries.num_queries);

  for (std::uint64_t query_id = 0; query_id < queries.num_queries; ++query_id) {
    std::vector<Neighbor> candidates(database.num_vectors);

    for (std::uint64_t vector_id = 0; vector_id < database.num_vectors;
         ++vector_id) {
      candidates[vector_id] = {
          vector_id,
          computeScore(database, queries, query_id, vector_id),
      };
    }
    results[query_id] =
        selectTopK(std::move(candidates), params.top_k, database.metric);
  }
  return results;
}

bool resultsMatch(const SearchResults& expected, const SearchResults& actual,
                  float absolute_tolerance, std::string* error_message) {
  if (expected.size() != actual.size()) {
    if (error_message != nullptr) {
      *error_message = "query 数量不同";
    }
    return false;
  }

  for (std::size_t query_id = 0; query_id < expected.size(); ++query_id) {
    if (expected[query_id].size() != actual[query_id].size()) {
      if (error_message != nullptr) {
        *error_message = "query " + std::to_string(query_id) + " 的 K 不同";
      }
      return false;
    }
    for (std::size_t rank = 0; rank < expected[query_id].size(); ++rank) {
      const Neighbor& left = expected[query_id][rank];
      const Neighbor& right = actual[query_id][rank];
      if (left.id != right.id ||
          std::fabs(left.score - right.score) > absolute_tolerance) {
        if (error_message != nullptr) {
          std::ostringstream message;
          message << "query=" << query_id << ", rank=" << rank
                  << " 不一致，CPU(id=" << left.id << ", score=" << left.score
                  << ")，GPU(id=" << right.id << ", score=" << right.score
                  << ')';
          *error_message = message.str();
        }
        return false;
      }
    }
  }
  return true;
}
