#pragma once

#include <string>

#include "vector_types.h"

VectorDatabase readVectorDatabase(const std::string& path);
QuerySet readQuerySet(const std::string& path);
MemoryMetadata readMemoryMetadata(const std::string& path);
void writeMemoryMetadata(const std::string& path,
                         const MemoryMetadata& metadata);
void validateMemoryMetadata(const MemoryMetadata& metadata,
                            std::uint64_t expected_vectors = 0);
SearchParams readSearchParams(const std::string& path);
void writeSearchResults(const std::string& path, const SearchResults& results);
void validateInputs(const VectorDatabase& database, const QuerySet& queries,
                    const SearchParams& params);
