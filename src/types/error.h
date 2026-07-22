// 错误码定义

#ifndef _ERROR_H_
#define _ERROR_H_

enum class Status
{
    SUCCESS,
    MODEL_NOT_FOUND,
    INIT_FAILED,
    LOAD_FAILED,
    INVALID_INPUT,
    INVALID_OUTPUT,
    RUNTIME_ERROR
};

#endif // RK3588_DEMO_ERROR_H
