/*!
 * @file    assert.hpp
 * @brief   assert macro used while executing
 */
#pragma once

#define RUNTIME_ASSERT(cond, message)                            \
    if (not(cond)) {                                             \
        std::cout << "[RUNTIME_ASSERT]" << message << std::endl; \
        exit(0);                                                 \
    }
