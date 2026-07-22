//
// Created for generic application usage.
//

#pragma once
#include <fmt/format.h>

#include <chrono>
#include <string>
#include <iostream>
#include <map>
#include <vector>
#include <memory>

#include "spdlog/sinks/stdout_color_sinks.h"
#ifndef __APPLE__
#include "spdlog/sinks/systemd_sink.h"
#endif
#include "spdlog/spdlog.h"

namespace core
{
    namespace log
    {

        class Logger
        {
        public:
            static Logger &Instance()
            {
                static Logger instance("runtime_logger");
                return instance;
            }

            Logger(std::string &&logger_name)
            {
                try
                {
                    std::vector<spdlog::sink_ptr> sinks;
                    if (IsEnableStdOut())
                    {
                        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
                        sinks.emplace_back(console_sink);
                    }
#ifndef __APPLE__
                    if (IsEnableSystemd())
                    {
                        auto systemd_sink = std::make_shared<spdlog::sinks::systemd_sink_mt>();
                        sinks.emplace_back(systemd_sink);
                    }
#endif

                    logger_ = std::make_shared<spdlog::logger>(logger_name, sinks.begin(), sinks.end());
                    logger_->set_level(ReadLogLevelFromEnv());
                    logger_->set_pattern("%v");
                }
                catch (const spdlog::spdlog_ex &ex)
                {
                    // 异常时保持静默或在此处处理
                }
            }

            ~Logger() {}
            const std::shared_ptr<spdlog::logger> GetLogger() const { return logger_; }

        private:
            spdlog::level::level_enum ReadLogLevelFromEnv()
            {
                const char *value = std::getenv("SPDLOG_LEVEL");
                if (value != nullptr)
                {
                    std::map<std::string, spdlog::level::level_enum> level_map = {
                        {"trace", spdlog::level::trace},
                        {"debug", spdlog::level::debug},
                        {"info", spdlog::level::info},
                        {"warning", spdlog::level::warn},
                        {"error", spdlog::level::err},
                        {"critical", spdlog::level::critical}};
                    try
                    {
                        return level_map.at(std::string(value));
                    }
                    catch (const std::out_of_range &e)
                    {
                        std::cout << "environment variable SPDLOG_LEVEL is invalid, and set default level\n";
                        return spdlog::level::info;
                    }
                }
                else
                {
                    std::cout << "environment variable SPDLOG_LEVEL is not set, and set default level\n";
                    return spdlog::level::info;
                }
            }

            bool IsEnableStdOut()
            {
                const char *value = std::getenv("SPDLOG_STDOUT");
                if (value != nullptr)
                {
                    std::string var = std::string(value);
                    if (var == "true")
                        return true;
                    if (var == "false")
                        return false;
                    std::cout << "environment variable SPDLOG_STDOUT is invalid, and set true\n";
                    return true;
                }
                std::cout << "environment variable SPDLOG_STDOUT is not set, and set true\n";
                return true;
            }

            bool IsEnableSystemd()
            {
                const char *value = std::getenv("SPDLOG_SYSTEMD");
                if (value != nullptr)
                {
                    std::string var = std::string(value);
                    if (var == "true")
                        return true;
                    if (var == "false")
                        return false;
                    std::cout << "environment variable SPDLOG_SYSTEMD is invalid, and set false\n";
                    return false;
                }
                std::cout << "environment variable SPDLOG_SYSTEMD is not set, and set false\n";
                return false;
            }

            std::shared_ptr<spdlog::logger> logger_;
        };

    } // namespace clog
}

#if SUFFIX_FORMAT_DISABLE
#define SUFFIX_FORMAT fmt::format("")
#else
#define SUFFIX_FORMAT fmt::format("{}() at {}:{}", __func__, __FILE__, __LINE__)
#endif
#define DEBUG_FORMAT (fmt::format("[DEBUG] "))
#define TRACE_FORMAT (fmt::format("[TRACE] "))
#define INFO_FORMAT (fmt::format("[INFO] "))
#define WARN_FORMAT (fmt::format("[WARN] "))
#define ERROR_FORMAT (fmt::format("[ERROR] "))
#define CRITICAL_FORMAT (fmt::format("[CRITICAL] "))

// ================= 全局宏定义接口 =================

#define LOG_TRACE(...)                                                \
    {                                                                 \
        SPDLOG_LOGGER_CALL(core::log::Logger::Instance().GetLogger(), \
                           spdlog::level::trace,                      \
                           TRACE_FORMAT + fmt::format(__VA_ARGS__));  \
    }
#define LOG_TRACE_ONCE(...)        \
    do                             \
    {                              \
        static bool once = true;   \
        if (once)                  \
        {                          \
            once = false;          \
            LOG_TRACE(__VA_ARGS__) \
        }                          \
    } while (0)

#define LOG_TRACE_FUNCTION(function, ...) \
    do                                    \
    {                                     \
        if ((*function)())                \
        {                                 \
            LOG_TRACE(__VA_ARGS__)        \
        }                                 \
    } while (0)

#define LOG_TRACE_EXPRESSION(expression, ...) \
    do                                        \
    {                                         \
        if (expression)                       \
        {                                     \
            LOG_TRACE(__VA_ARGS__)            \
        }                                     \
    } while (0)

#undef LOG_DEBUG
#define LOG_DEBUG(...)                                                \
    {                                                                 \
        SPDLOG_LOGGER_CALL(core::log::Logger::Instance().GetLogger(), \
                           spdlog::level::debug,                      \
                           DEBUG_FORMAT + fmt::format(__VA_ARGS__));  \
    }
#define LOG_DEBUG_ONCE(...)        \
    do                             \
    {                              \
        static bool once = true;   \
        if (once)                  \
        {                          \
            once = false;          \
            LOG_DEBUG(__VA_ARGS__) \
        }                          \
    } while (0)

#define LOG_DEBUG_FUNCTION(function, ...) \
    do                                    \
    {                                     \
        if ((*function)())                \
        {                                 \
            LOG_DEBUG(__VA_ARGS__)        \
        }                                 \
    } while (0)

#define LOG_DEBUG_EXPRESSION(expression, ...) \
    do                                        \
    {                                         \
        if (expression)                       \
        {                                     \
            LOG_DEBUG(__VA_ARGS__)            \
        }                                     \
    } while (0)

#undef LOG_INFO
#define LOG_INFO(...)                                                 \
    {                                                                 \
        SPDLOG_LOGGER_CALL(core::log::Logger::Instance().GetLogger(), \
                           spdlog::level::info,                       \
                           INFO_FORMAT + fmt::format(__VA_ARGS__));   \
    }
#define LOG_INFO_ONCE(...)        \
    do                            \
    {                             \
        static bool once = true;  \
        if (once)                 \
        {                         \
            once = false;         \
            LOG_INFO(__VA_ARGS__) \
        }                         \
    } while (0)

#define LOG_INFO_FUNCTION(function, ...) \
    do                                   \
    {                                    \
        if ((*function)())               \
        {                                \
            LOG_INFO(__VA_ARGS__)        \
        }                                \
    } while (0)

#define LOG_INFO_EXPRESSION(expression, ...) \
    do                                       \
    {                                        \
        if (expression)                      \
        {                                    \
            LOG_INFO(__VA_ARGS__)            \
        }                                    \
    } while (0)

#define LOG_WARN(...)                                                 \
    {                                                                 \
        SPDLOG_LOGGER_CALL(core::log::Logger::Instance().GetLogger(), \
                           spdlog::level::warn,                       \
                           WARN_FORMAT + fmt::format(__VA_ARGS__));   \
    }
#define LOG_WARN_ONCE(...)        \
    do                            \
    {                             \
        static bool once = true;  \
        if (once)                 \
        {                         \
            once = false;         \
            LOG_WARN(__VA_ARGS__) \
        }                         \
    } while (0)

#define LOG_WARN_FUNCTION(function, ...) \
    do                                   \
    {                                    \
        if ((*function)())               \
        {                                \
            LOG_WARN(__VA_ARGS__)        \
        }                                \
    } while (0)

#define LOG_WARN_EXPRESSION(expression, ...) \
    do                                       \
    {                                        \
        if (expression)                      \
        {                                    \
            LOG_WARN(__VA_ARGS__)            \
        }                                    \
    } while (0)

#define LOG_ERROR(...)                                                \
    {                                                                 \
        SPDLOG_LOGGER_CALL(core::log::Logger::Instance().GetLogger(), \
                           spdlog::level::err,                        \
                           ERROR_FORMAT + fmt::format(__VA_ARGS__));  \
    }
#define LOG_ERROR_ONCE(...)        \
    do                             \
    {                              \
        static bool once = true;   \
        if (once)                  \
        {                          \
            once = false;          \
            LOG_ERROR(__VA_ARGS__) \
        }                          \
    } while (0)

#define LOG_ERROR_FUNCTION(function, ...) \
    do                                    \
    {                                     \
        if ((*function)())                \
        {                                 \
            LOG_ERROR(__VA_ARGS__)        \
        }                                 \
    } while (0)

#define LOG_ERROR_EXPRESSION(expression, ...) \
    do                                        \
    {                                         \
        if (expression)                       \
        {                                     \
            LOG_ERROR(__VA_ARGS__)            \
        }                                     \
    } while (0)

#define LOG_CRITICAL(...)                                               \
    {                                                                   \
        SPDLOG_LOGGER_CALL(core::log::Logger::Instance().GetLogger(),   \
                           spdlog::level::critical,                     \
                           CRITICAL_FORMAT + fmt::format(__VA_ARGS__)); \
    }
#define LOG_CRITICAL_ONCE(...)        \
    do                                \
    {                                 \
        static bool once = true;      \
        if (once)                     \
        {                             \
            once = false;             \
            LOG_CRITICAL(__VA_ARGS__) \
        }                             \
    } while (0)

#define LOG_CRITICAL_FUNCTION(function, ...) \
    do                                       \
    {                                        \
        if ((*function)())                   \
        {                                    \
            LOG_CRITICAL(__VA_ARGS__)        \
        }                                    \
    } while (0)

#define LOG_CRITICAL_EXPRESSION(expression, ...) \
    do                                           \
    {                                            \
        if (expression)                          \
        {                                        \
            LOG_CRITICAL(__VA_ARGS__)            \
        }                                        \
    } while (0)

#define TIME_COST_INFO(description, codeBlock)                                 \
    do                                                                         \
    {                                                                          \
        auto start = std::chrono::high_resolution_clock::now();                \
        codeBlock;                                                             \
        auto end = std::chrono::high_resolution_clock::now();                  \
        auto duration =                                                        \
            std::chrono::duration_cast<std::chrono::microseconds>(end - start) \
                .count() /                                                     \
            1000.0f;                                                           \
        LOG_INFO("{} cost time: {}ms", (description), (duration));             \
    } while (0)

#define TIME_COST_DEBUG(description, codeBlock)                                \
    do                                                                         \
    {                                                                          \
        auto start = std::chrono::high_resolution_clock::now();                \
        codeBlock;                                                             \
        auto end = std::chrono::high_resolution_clock::now();                  \
        auto duration =                                                        \
            std::chrono::duration_cast<std::chrono::microseconds>(end - start) \
                .count() /                                                     \
            1000.0f;                                                           \
        LOG_DEBUG("{} cost time: {}ms", (description), (duration));            \
    } while (0)

#define LOG_ERROR_CODE(err_code, ...)                                      \
    {                                                                      \
        SPDLOG_LOGGER_CALL(                                                \
            core::log::Logger::Instance().GetLogger(), spdlog::level::err, \
            ERROR_FORMAT + fmt::format("<ERROR_CODE: {}> ", (err_code)) +  \
                fmt::format(__VA_ARGS__));                                 \
    }
