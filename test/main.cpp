#include <cassert>
#include <cstdio>
#include <cstring>

#include "engine/ai_instance.h"

int main()
{
    auto instance = ai_framework::Engine(ai_framework::RKNN_FORMAT,
                                         "/home/orangepi/Code/Openmmlab/model/rtmpose-m_8xb256_hand_finetune_pow-fp16.rknn");
    return 0;
}
