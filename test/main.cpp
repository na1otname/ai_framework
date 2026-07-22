#include <cassert>
#include <cstdio>
#include <cstring>

#include "engine/ai_instance.h"

int main()
{
    auto instance = ai_framework::Engine(ai_framework::RKNN_FORMAT,
                                         "/home/orangepi/Code/ai_framework/model/rtmdet_nano_320x320-fp16.rknn");
    return 0;
}
