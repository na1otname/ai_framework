#ifndef _IMAGE_H_
#define _IMAGE_H_

class Image
{
public:
    int width;
    int height;
    int channel;
    void* data;
};

#endif