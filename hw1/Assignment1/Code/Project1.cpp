#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include <iostream>
#include <vector>
#include <algorithm>

/***********************************************************************
  This is the only file you need to change for your assignment. The
  other files control the UI (in case you want to make changes.)
************************************************************************/

/***********************************************************************
  The first eight functions provide example code to help get you started
************************************************************************/


// Convert an image to grayscale
void MainWindow::BlackWhiteImage(QImage *image)
{
    for(int r=0;r<image->height();r++)
        for(int c=0;c<image->width();c++)
        {
            QRgb pixel = image->pixel(c, r);
            double red = (double) qRed(pixel);
            double green = (double) qGreen(pixel);
            double blue = (double) qBlue(pixel);

            // Compute intensity from colors - these are common weights
            double intensity = 0.3*red + 0.6*green + 0.1*blue;

            image->setPixel(c, r, qRgb( (int) intensity, (int) intensity, (int) intensity));
        }
}

// Add random noise to the image
void MainWindow::AddNoise(QImage *image, double mag, bool colorNoise)
{
    int noiseMag = mag*2;

    for(int r=0;r<image->height();r++)
        for(int c=0;c<image->width();c++)
        {
            QRgb pixel = image->pixel(c, r);
            int red = qRed(pixel), green = qGreen(pixel), blue = qBlue(pixel);

            // If colorNoise, add color independently to each channel
            if(colorNoise)
            {
                red += rand()%noiseMag - noiseMag/2;
                green += rand()%noiseMag - noiseMag/2;
                blue += rand()%noiseMag - noiseMag/2;
            }
            // otherwise add the same amount of noise to each channel
            else
            {
                int noise = rand()%noiseMag - noiseMag/2;
                red += noise; green += noise; blue += noise;
            }
            image->setPixel(c, r, qRgb(max(0, min(255, red)), max(0, min(255, green)), max(0, min(255, blue))));
        }
}

// Downsample the image by 1/2
void MainWindow::HalfImage(QImage &image)
{
    int w = image.width();
    int h = image.height();
    QImage buffer = image.copy();

    // Reduce the image size.
    image = QImage(w/2, h/2, QImage::Format_RGB32);

    // Copy every other pixel
    for(int r=0;r<h/2;r++)
        for(int c=0;c<w/2;c++)
             image.setPixel(c, r, buffer.pixel(c*2, r*2));
}

// Round float values to the nearest integer values and make sure the value lies in the range [0,255]
QRgb restrictColor(double red, double green, double blue)
{
    int r = (int)(floor(red+0.5));
    int g = (int)(floor(green+0.5));
    int b = (int)(floor(blue+0.5));

    return qRgb(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)));
}

// Normalize the values of the kernel to sum-to-one
void NormalizeKernel(double *kernel, int kernelWidth, int kernelHeight)
{
    double denom = 0.000001; int i;
    for(i=0; i<kernelWidth*kernelHeight; i++)
        denom += kernel[i];
    for(i=0; i<kernelWidth*kernelHeight; i++)
        kernel[i] /= denom;
}

// Here is an example of blurring an image using a mean or box filter with the specified radius.
// This could be implemented using separable filters to make it much more efficient, but it's not done here.
// Note: This function is written using QImage form of the input image. But all other functions later use the double form
void MainWindow::MeanBlurImage(QImage *image, int radius)
{
    std::cout<<"hi"<<std::endl;
    if(radius == 0)
        return;
    int size = 2*radius + 1; // This is the size of the kernel

    // Note: You can access the width and height using 'imageWidth' and 'imageHeight' respectively in the functions you write
    int w = image->width();
    int h = image->height();

    // Create a buffer image so we're not reading and writing to the same image during filtering.
    // This creates an image of size (w + 2*radius, h + 2*radius) with black borders (zero-padding)
    QImage buffer = image->copy(-radius, -radius, w + 2*radius, h + 2*radius);

    // Compute the kernel to convolve with the image
    double *kernel = new double [size*size];

    for(int i=0;i<size*size;i++)
        kernel[i] = 1.0;

    // Make sure kernel sums to 1
    NormalizeKernel(kernel, size, size);

    // For each pixel in the image...
    for(int r=0;r<h;r++)
    {
        for(int c=0;c<w;c++)
        {
            double rgb[3];
            rgb[0] = rgb[1] = rgb[2] = 0.0;

            // Convolve the kernel at each pixel
            for(int rd=-radius;rd<=radius;rd++)
                for(int cd=-radius;cd<=radius;cd++)
                {
                     // Get the pixel value
                     //For the functions you write, check the ConvertQImage2Double function to see how to get the pixel value
                     QRgb pixel = buffer.pixel(c + cd + radius, r + rd + radius);

                     // Get the value of the kernel
                     double weight = kernel[(rd + radius)*size + cd + radius];

                     rgb[0] += weight*(double) qRed(pixel);
                     rgb[1] += weight*(double) qGreen(pixel);
                     rgb[2] += weight*(double) qBlue(pixel);
                }
            // Store the pixel in the image to be returned
            // You need to store the RGB values in the double form of the image
            image->setPixel(c, r, restrictColor(rgb[0],rgb[1],rgb[2]));
        }
    }
    // Clean up (use this carefully)
    delete[] kernel;
}

// Convert QImage to a matrix of size (imageWidth*imageHeight)*3 having double values
void MainWindow::ConvertQImage2Double(QImage image)
{
    // Global variables to access image width and height
    imageWidth = image.width();
    imageHeight = image.height();

    // Initialize the global matrix holding the image
    // This is how you will be creating a copy of the original image inside a function
    // Note: 'Image' is of type 'double**' and is declared in the header file (hence global variable)
    // So, when you create a copy (say buffer), write "double** buffer = new double ....."
    Image = new double* [imageWidth*imageHeight];
    for (int i = 0; i < imageWidth*imageHeight; i++)
            Image[i] = new double[3];

    // For each pixel
    for (int r = 0; r < imageHeight; r++)
        for (int c = 0; c < imageWidth; c++)
        {
            // Get a pixel from the QImage form of the image
            QRgb pixel = image.pixel(c,r);

            // Assign the red, green and blue channel values to the 0, 1 and 2 channels of the double form of the image respectively
            Image[r*imageWidth+c][0] = (double) qRed(pixel);
            Image[r*imageWidth+c][1] = (double) qGreen(pixel);
            Image[r*imageWidth+c][2] = (double) qBlue(pixel);
        }
}

// Convert the matrix form of the image back to QImage for display
void MainWindow::ConvertDouble2QImage(QImage *image)
{
    for (int r = 0; r < imageHeight; r++)
        for (int c = 0; c < imageWidth; c++)
            image->setPixel(c, r, restrictColor(Image[r*imageWidth+c][0], Image[r*imageWidth+c][1], Image[r*imageWidth+c][2]));
}


/**************************************************
 TIME TO WRITE CODE
**************************************************/

/**************************************************
 TASK 1
**************************************************/

// Convolve the image with the kernel
void MainWindow::Convolution(double** image, double *kernel, int kernelWidth, int kernelHeight, bool add)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * kernel: 1-D array of kernel values
 * kernelWidth: width of the kernel
 * kernelHeight: height of the kernel
 * add: a boolean variable (taking values true or false)
*/
{
    // Add your code here

    // create buffer using dynamic 2D array
    int buffer_w = imageWidth + kernelWidth - 1;
    int buffer_h = imageHeight + kernelHeight - 1;
    int w = imageWidth;
    int h = imageHeight;
    double** buffer = new double*[buffer_w * buffer_h];
    for(int i = 0; i < buffer_w * buffer_h; ++i)
        buffer[i] = new double[3];

    // if useZeroPdding is set to true, use zero padding, use fixed padding otherwise
    bool useZeroPadding = true;
    // assign image to buffer also add zero padding
    if(useZeroPadding){
        for(int i=0; i<buffer_h; i++){
            for(int j=0; j<buffer_w; j++){
                if((i<kernelHeight/2)||(i>=imageHeight+kernelHeight/2)||(j<kernelWidth/2)||(j>=imageWidth+kernelWidth/2)){
                    buffer[i * buffer_w + j][0] = 0.0;
                    buffer[i * buffer_w + j][1] = 0.0;
                    buffer[i * buffer_w + j][2] = 0.0;
                }
                else{
                    buffer[i * buffer_w + j][0] = image[(i-kernelHeight/2)*imageWidth + (j-kernelWidth/2)][0];
                    buffer[i * buffer_w + j][1] = image[(i-kernelHeight/2)*imageWidth + (j-kernelWidth/2)][1];
                    buffer[i * buffer_w + j][2] = image[(i-kernelHeight/2)*imageWidth + (j-kernelWidth/2)][2];
                }
            }
        }
    }


    // assign image to buffer but use fixed padding
    else{
        for(int i=0; i<buffer_h; i++){
            for(int j=0; j<buffer_w; j++){
                if((i<kernelHeight/2) && (j<kernelWidth/2)){
                    buffer[i * buffer_w + j][0] = image[0][0];
                    buffer[i * buffer_w + j][1] = image[0][1];
                    buffer[i * buffer_w + j][2] = image[0][2];
                }
                else if((i<kernelHeight/2) && (j>=imageWidth+kernelWidth/2)){
                    buffer[i * buffer_w + j][0] = image[imageWidth-1][0];
                    buffer[i * buffer_w + j][1] = image[imageWidth-1][1];
                    buffer[i * buffer_w + j][2] = image[imageWidth-1][2];
                }
                else if((j<kernelWidth/2) && (i>=imageHeight+kernelHeight/2)){
                    buffer[i * buffer_w + j][0] = image[(imageHeight-1)*imageWidth][0];
                    buffer[i * buffer_w + j][1] = image[(imageHeight-1)*imageWidth][1];
                    buffer[i * buffer_w + j][2] = image[(imageHeight-1)*imageWidth][2];
                }
                else if((j>=imageWidth+kernelWidth/2) && (i>=imageHeight+kernelHeight/2)){
                    buffer[i * buffer_w + j][0] = image[imageHeight*imageWidth-1][0];
                    buffer[i * buffer_w + j][1] = image[imageHeight*imageWidth-1][1];
                    buffer[i * buffer_w + j][2] = image[imageHeight*imageWidth-1][2];
                }
                else if((i<kernelHeight/2)){
                    buffer[i * buffer_w + j][0] = image[0*imageWidth + (j-kernelWidth/2)][0];
                    buffer[i * buffer_w + j][1] = image[0*imageWidth + (j-kernelWidth/2)][1];
                    buffer[i * buffer_w + j][2] = image[0*imageWidth + (j-kernelWidth/2)][2];
                }
                else if((j<kernelWidth/2)){
                    buffer[i * buffer_w + j][0] = image[(i-kernelHeight/2)*imageWidth + 0][0];
                    buffer[i * buffer_w + j][1] = image[(i-kernelHeight/2)*imageWidth + 0][1];
                    buffer[i * buffer_w + j][2] = image[(i-kernelHeight/2)*imageWidth + 0][2];
                }
                else if((i>=imageHeight+kernelHeight/2)){
                    buffer[i * buffer_w + j][0] = image[(imageHeight-1)*imageWidth + (j-kernelWidth/2)][0];
                    buffer[i * buffer_w + j][1] = image[(imageHeight-1)*imageWidth + (j-kernelWidth/2)][1];
                    buffer[i * buffer_w + j][2] = image[(imageHeight-1)*imageWidth + (j-kernelWidth/2)][2];
                }
                else if((j>=imageWidth+kernelWidth/2)){
                    buffer[i * buffer_w + j][0] = image[(i-kernelHeight/2)*imageWidth + (imageWidth-1)][0];
                    buffer[i * buffer_w + j][1] = image[(i-kernelHeight/2)*imageWidth + (imageWidth-1)][1];
                    buffer[i * buffer_w + j][2] = image[(i-kernelHeight/2)*imageWidth + (imageWidth-1)][2];
                }
                else{
                    buffer[i * buffer_w + j][0] = image[(i-kernelHeight/2)*imageWidth + (j-kernelWidth/2)][0];
                    buffer[i * buffer_w + j][1] = image[(i-kernelHeight/2)*imageWidth + (j-kernelWidth/2)][1];
                    buffer[i * buffer_w + j][2] = image[(i-kernelHeight/2)*imageWidth + (j-kernelWidth/2)][2];
                }
            }
        }
    }


    // print original image (for debugging)
//    for(int i=0; i<h; i++){
//        for(int j=0; j<w; j++){
//            std::cout<<image[i * w + j][0]<<" ";
//        }
//        std::cout<<std::endl;
//    }
    // print out buffer image (after padding) (for debugging)
//    for(int i=0; i<buffer_h; i++){
//        for(int j=0; j<buffer_w; j++){
//            std::cout<<buffer[i * buffer_w + j][0]<<" ";
//        }
//        std::cout<<std::endl;
//    }

    // do the convolution
    for(int r=kernelHeight/2;r<h+kernelHeight/2;r++)
    {
        for(int c=kernelWidth/2;c<w+kernelWidth/2;c++)
        {
            double rgb[3];
            rgb[0] = rgb[1] = rgb[2] = 0.0;

            // Convolve the kernel at each pixel
            for(int rd = -kernelHeight/2; rd <= kernelHeight/2; rd++)
            {
                for(int cd = -kernelWidth/2; cd <= kernelWidth/2; cd++)
                {
                     double* pixel = buffer[(r+rd) * buffer_w + (c+cd)];

                     // Get the value of the kernel
                     double weight = kernel[(rd + (kernelHeight/2))*kernelWidth + cd + (kernelWidth/2)];

                     rgb[0] += weight*(double) pixel[0];
                     rgb[1] += weight*(double) pixel[1];
                     rgb[2] += weight*(double) pixel[2];
                }
            }
            // Store the pixel in the image to be returned
            // You need to store the RGB values in the double form of the image
            if(!add)
            {
                image[(r-kernelHeight/2) * w + (c-kernelWidth/2)][0] = rgb[0];
                image[(r-kernelHeight/2) * w + (c-kernelWidth/2)][1] = rgb[1];
                image[(r-kernelHeight/2) * w + (c-kernelWidth/2)][2] = rgb[2];
            }
            else
            {
                image[(r-kernelHeight/2) * w + (c-kernelWidth/2)][0] = rgb[0]+128;
                image[(r-kernelHeight/2) * w + (c-kernelWidth/2)][1] = rgb[1]+128;
                image[(r-kernelHeight/2) * w + (c-kernelWidth/2)][2] = rgb[2]+128;
            }
        }
    }

    // delete dynamic allocated memory
    for(int i = 0; i < buffer_w * buffer_h; ++i) {
        delete [] buffer[i];
    }
    delete [] buffer;
}

/**************************************************
 TASK 2
**************************************************/

// Apply the 2-D Gaussian kernel on an image to blur it
void MainWindow::GaussianBlurImage(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
    // Add your code here
    if(sigma == 0.0)
        return;
    int radius = (int)(ceil(3*sigma));
    int size = 2*radius + 1; // This is the size of the kernel
    double r, s = 2.0 * sigma * sigma;

    // Compute the kernel to convolve with the image
    double *kernel = new double [size*size];
    for(int x = -radius; x <= radius; x++){
        for(int y = -radius; y <= radius; y++){
            r = sqrt(x * x + y * y);
            kernel[(x+radius) * size + (y+radius)] = (exp(-(r * r) / s)) / (M_PI * s);
        }
    }

    // Make sure kernel sums to 1
    NormalizeKernel(kernel, size, size);

    // print gaussian kernel (for debugging)
//    for(int i=0; i<size; i++){
//        for(int j=0; j<size; j++){
//            std::cout<<kernel[i * size + j]<<" ";
//        }
//        std::cout<<std::endl;
//    }

    // Do convolution
    Convolution(image, kernel, size, size, false);

    // Clean up
    delete[] kernel;

}

/**************************************************
 TASK 3
**************************************************/

// Perform the Gaussian Blur first in the horizontal direction and then in the vertical direction
void MainWindow::SeparableGaussianBlurImage(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
    // Add your code here
    if(sigma == 0.0)
        return;
    int radius = (int)(ceil(3*sigma));
    int size = 2*radius + 1; // This is the size of the kernel
    double r, s = 2.0 * sigma * sigma;

    // Compute the horizontal kernel to convolve with the image
    double *kernel = new double [size*size];
    for(int x = -radius; x <= radius; x++){
        for(int y = -radius; y <= radius; y++){
            if(x==0){
                r = sqrt(x * x + y * y);
                kernel[(x+radius) * size + (y+radius)] = (exp(-(r * r) / s)) / (M_PI * s);
            }
            else{
                kernel[(x+radius) * size + (y+radius)] = 0.0;
            }

        }
    }

    // Make sure kernel sums to 1
    NormalizeKernel(kernel, size, size);

    // print gaussian kernel (for debugging)
//    for(int i=0; i<size; i++){
//        for(int j=0; j<size; j++){
//            std::cout<<kernel[i * size + j]<<" ";
//        }
//        std::cout<<std::endl;
//    }

    // Do convolution
    Convolution(image, kernel, size, size, false);

    // Compute the vertical kernel to convolve with the image
    for(int x = -radius; x <= radius; x++){
        for(int y = -radius; y <= radius; y++){
            if(y==0){
                r = sqrt(x * x + y * y);
                kernel[(x+radius) * size + (y+radius)] = (exp(-(r * r) / s)) / (M_PI * s);
            }
            else{
                kernel[(x+radius) * size + (y+radius)] = 0.0;
            }

        }
    }

    // Make sure kernel sums to 1
    NormalizeKernel(kernel, size, size);

    // print gaussian kernel (for debugging)
//    for(int i=0; i<size; i++){
//        for(int j=0; j<size; j++){
//            std::cout<<kernel[i * size + j]<<" ";
//        }
//        std::cout<<std::endl;
//    }

    // Do convolution
    Convolution(image, kernel, size, size, false);

    // Clean up
    delete[] kernel;
}

/********** TASK 4 (a) **********/

// Compute the First derivative of an image along the horizontal direction and then apply Gaussian blur.
void MainWindow::FirstDerivImage_x(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
    // Add your code here

    // Compute the horizontal first derivative to convolve with the image
    double *kernel = new double [3*3];
    for(int x = -1; x <= 1; x++){
        for(int y = -1; y <= 1; y++){
            if(x==0){
                if(y == -1){
                    kernel[(x+1) * 3 + (y+1)] = -1;
                }
                else if(y == 1) {
                    kernel[(x+1) * 3 + (y+1)] = 1;
                }
                else{
                    kernel[(x+1) * 3 + (y+1)] = 0.0;
                }
            }
            else{
                kernel[(x+1) * 3 + (y+1)] = 0.0;
            }
        }
    }
    Convolution(image, kernel, 3, 3, true);
    GaussianBlurImage(image, sigma);
    // Clean up
    delete[] kernel;
}

/********** TASK 4 (b) **********/

// Compute the First derivative of an image along the vertical direction and then apply Gaussian blur.
void MainWindow::FirstDerivImage_y(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
    // Add your code here
    // Compute the vertical first derivative to convolve with the image
    double *kernel = new double [3*3];
    for(int x = -1; x <= 1; x++){
        for(int y = -1; y <= 1; y++){
            if(y==0){
                if(x == -1){
                    kernel[(x+1) * 3 + (y+1)] = 1;
                }
                else if(x == 1) {
                    kernel[(x+1) * 3 + (y+1)] = -1;
                }
                else{
                    kernel[(x+1) * 3 + (y+1)] = 0.0;
                }
            }
            else{
                kernel[(x+1) * 3 + (y+1)] = 0.0;
            }
        }
    }
    Convolution(image, kernel, 3, 3, true);
    GaussianBlurImage(image, sigma);
    // Clean up
    delete[] kernel;
}

/********** TASK 4 (c) **********/

// Compute the Second derivative of an image using the Laplacian operator and then apply Gaussian blur
void MainWindow::SecondDerivImage(double** image, double sigma)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
*/
{
    // Add your code here
    // Compute the second derivative to convolve with the image
    double *kernel = new double [3*3];
    for(int x = -1; x <= 1; x++){
        for(int y = -1; y <= 1; y++){
            if(y==0 || x==0){
                    kernel[(x+1) * 3 + (y+1)] = 1.0;
            }
            else{
                kernel[(x+1) * 3 + (y+1)] = 0.0;
            }
        }
    }
    kernel[4] = -4.0;
    Convolution(image, kernel, 3, 3, true);
    GaussianBlurImage(image, sigma);
    // Clean up
    delete[] kernel;
}

/**************************************************
 TASK 5
**************************************************/

// Sharpen an image by subtracting the image's second derivative from the original image
void MainWindow::SharpenImage(double** image, double sigma, double alpha)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigma: standard deviation for the Gaussian kernel
 * alpha: constant by which the second derivative image is to be multiplied to before subtracting it from the original image
*/
{
    // Add your code here
    int size = imageWidth * imageHeight;
    double** buffer = new double*[size];
    for(int i = 0; i < size; ++i)
        buffer[i] = new double[3];

    // assign image to buffer
    for(int i=0; i<imageHeight; i++){
        for(int j=0; j<imageWidth; j++){
            buffer[i * imageWidth + j][0] = image[i * imageWidth + j][0];
            buffer[i * imageWidth + j][1] = image[i * imageWidth + j][1];
            buffer[i * imageWidth + j][2] = image[i * imageWidth + j][2];
        }
    }
    SecondDerivImage(buffer, sigma);
    for(int i=0; i<imageHeight; i++){
        for(int j=0; j<imageWidth; j++){
            image[i * imageWidth + j][0] = image[i * imageWidth + j][0] - alpha*(buffer[i * imageWidth + j][0] - 128);
            image[i * imageWidth + j][1] = image[i * imageWidth + j][1] - alpha*(buffer[i * imageWidth + j][1] - 128);
            image[i * imageWidth + j][2] = image[i * imageWidth + j][2] - alpha*(buffer[i * imageWidth + j][2] - 128);
        }
    }

    // delete resource
    for(int i = 0; i < size; ++i) {
        delete [] buffer[i];
    }
    delete [] buffer;
}

/**************************************************
 TASK 6
**************************************************/

// Display the magnitude and orientation of the edges in an image using the Sobel operator in both X and Y directions
void MainWindow::SobelImage(double** image)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * NOTE: image is grayscale here, i.e., all 3 channels have the same value which is the grayscale value
*/
{
    // Add your code here

    // create first buffer for vertical
    int size = imageWidth * imageHeight;
    double** buffer1 = new double*[size];
    for(int i = 0; i < size; ++i)
        buffer1[i] = new double[3];
    for(int i=0; i<imageHeight; i++){
        for(int j=0; j<imageWidth; j++){
            buffer1[i * imageWidth + j][0] = image[i * imageWidth + j][0];
            buffer1[i * imageWidth + j][1] = image[i * imageWidth + j][1];
            buffer1[i * imageWidth + j][2] = image[i * imageWidth + j][2];
        }
    }
    // create second buffer for vertical
    double** buffer2 = new double*[size];
    for(int i = 0; i < size; ++i)
        buffer2[i] = new double[3];
    for(int i=0; i<imageHeight; i++){
        for(int j=0; j<imageWidth; j++){
            buffer2[i * imageWidth + j][0] = image[i * imageWidth + j][0];
            buffer2[i * imageWidth + j][1] = image[i * imageWidth + j][1];
            buffer2[i * imageWidth + j][2] = image[i * imageWidth + j][2];
        }
    }

    // create kernel
    double* kernel = new double[3*3];
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            if(j==2){
                if(i==1) kernel[i*3+j] = 2;
                else kernel[i*3+j] = 1;
            }
            else if(j==1){
                kernel[i*3+j] = 0.0;
            }
            else{
                if(i==1) kernel[i*3+j] = -2;
                else kernel[i*3+j] = -1;
            }
        }
    }

    Convolution(buffer1, kernel, 3, 3, false);

    // create another kernel (using the same kernel memory)
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            if(i==2){
                if(j==1) kernel[i*3+j] = -2;
                else kernel[i*3+j] = -1;
            }
            else if(i==1){
                kernel[i*3+j] = 0.0;
            }
            else{
                if(j==1) kernel[i*3+j] = 2;
                else kernel[i*3+j] = 1;
            }
        }
    }
    Convolution(buffer2, kernel, 3, 3, false);
    double mag, orien;
    for(int r=0; r<imageHeight; r++){
        for(int c=0; c<imageWidth; c++){
            mag = sqrt(buffer1[r * imageWidth + c][0]*buffer1[r * imageWidth + c][0]/64.0 + buffer2[r * imageWidth + c][0]*buffer2[r * imageWidth + c][0]/64.0);
            orien = atan2(buffer2[r * imageWidth + c][0]/8.0, buffer1[r * imageWidth + c][0]/8.0);
            image[r * imageWidth + c][0] = mag*4.0*((sin(orien) + 1.0)/2.0);
            image[r * imageWidth + c][1] = mag*4.0*((cos(orien) + 1.0)/2.0);
            image[r * imageWidth + c][2] = mag*4.0 - image[r*imageWidth+c][0] - image[r*imageWidth+c][1];
        }
    }

    // delete resources
    for(int i = 0; i < size; ++i) {
        delete [] buffer1[i];
        delete [] buffer2[i];
    }
    delete [] buffer1;
    delete [] buffer2;
    delete [] kernel;

    // Use the following 3 lines of code to set the image pixel values after computing magnitude and orientation
    // Here 'mag' is the magnitude and 'orien' is the orientation angle in radians to be computed using atan2 function
    // (sin(orien) + 1)/2 converts the sine value to the range [0,1]. Similarly for cosine.

    // image[r*imageWidth+c][0] = mag*4.0*((sin(orien) + 1.0)/2.0);
    // image[r*imageWidth+c][1] = mag*4.0*((cos(orien) + 1.0)/2.0);
    // image[r*imageWidth+c][2] = mag*4.0 - image[r*imageWidth+c][0] - image[r*imageWidth+c][1];
}

/**************************************************
 TASK 7
**************************************************/

// Compute the RGB values at a given point in an image using bilinear interpolation.
void MainWindow::BilinearInterpolation(double** image, double x, double y, double rgb[3])
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * x: x-coordinate (corresponding to columns) of the position whose RGB values are to be found
 * y: y-coordinate (corresponding to rows) of the position whose RGB values are to be found
 * rgb[3]: array where the computed RGB values are to be stored
*/
{
    int x1, x2, y1, y2;
    double x11, x22, y11, y22;
    // Add your code here

    // check the boundary
    if(x < 0 || y<0 || x >= imageWidth || y >= imageHeight){
        rgb[0] = 0.0; rgb[1] = 0.0; rgb[2] = 0.0;
    }
    else
    {
        // also check the boundary
        x2 = (int)(ceil(x))>=imageWidth? imageWidth-1: (int)(ceil(x));
        x1 = x2 - 1<0? 0: x2 - 1;
        y2 = (int)(ceil(y))>=imageHeight? imageHeight-1: (int)(ceil(y));
        y1 = y2 - 1<0? 0: y2 - 1;
        x11 = (double)x1;
        x22 = (double)x2;
        y11 = (double)y1;
        y22 = (double)y2;


        double area = (x22 - x11) * (y22 - y11);

        // use the formula described in hw instruction
        rgb[0] = (1.0/area) * ((image[y1*imageWidth+x1][0]*(x22-x)*(y22-y)) + (image[y2*imageWidth+x1][0]*(x-x11)*(y22-y)) + (image[y1*imageWidth+x2][0]*(x22-x)*(y-y11)) + (image[y2*imageWidth+x2][0]*(x-x11)*(y-y11)));
        rgb[1] = (1.0/area) * ((image[y1*imageWidth+x1][1]*(x22-x)*(y22-y)) + (image[y2*imageWidth+x1][1]*(x-x11)*(y22-y)) + (image[y1*imageWidth+x2][1]*(x22-x)*(y-y11)) + (image[y2*imageWidth+x2][1]*(x-x11)*(y-y11)));
        rgb[2] = (1.0/area) * ((image[y1*imageWidth+x1][2]*(x22-x)*(y22-y)) + (image[y2*imageWidth+x1][2]*(x-x11)*(y22-y)) + (image[y1*imageWidth+x2][2]*(x22-x)*(y-y11)) + (image[y2*imageWidth+x2][2]*(x-x11)*(y-y11)));
    }

}

/*******************************************************************************
 Here is the code provided for rotating an image. 'orien' is in degrees.
********************************************************************************/

// Rotating an image by "orien" degrees
void MainWindow::RotateImage(double** image, double orien)

{
    double radians = -2.0*3.141*orien/360.0;

    // Make a copy of the original image and then re-initialize the original image with 0
    double** buffer = new double* [imageWidth*imageHeight];
    for (int i = 0; i < imageWidth*imageHeight; i++)
    {
        buffer[i] = new double [3];
        for(int j = 0; j < 3; j++)
            buffer[i][j] = image[i][j];
        image[i] = new double [3](); // re-initialize to 0
    }

    for (int r = 0; r < imageHeight; r++)
       for (int c = 0; c < imageWidth; c++)
       {
            // Rotate around the center of the image
            double x0 = (double) (c - imageWidth/2);
            double y0 = (double) (r - imageHeight/2);

            // Rotate using rotation matrix
            double x1 = x0*cos(radians) - y0*sin(radians);
            double y1 = x0*sin(radians) + y0*cos(radians);

            x1 += (double) (imageWidth/2);
            y1 += (double) (imageHeight/2);

            double rgb[3];
            BilinearInterpolation(buffer, x1, y1, rgb);

            // Note: "image[r*imageWidth+c] = rgb" merely copies the head pointers of the arrays, not the values
            image[r*imageWidth+c][0] = rgb[0];
            image[r*imageWidth+c][1] = rgb[1];
            image[r*imageWidth+c][2] = rgb[2];
        }
    // add by myself
    for(int i = 0; i < imageWidth*imageHeight; ++i) {
        delete [] buffer[i];
    }
    delete [] buffer;
}

/**************************************************
 TASK 8
**************************************************/

// Find the peaks of the edge responses perpendicular to the edges
void MainWindow::FindPeaksImage(double** image, double thres)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * NOTE: image is grayscale here, i.e., all 3 channels have the same value which is the grayscale value
 * thres: threshold value for magnitude
*/
{
    // Add your code here
    // create first buffer for vertical
    int size = imageWidth * imageHeight;
    double** buffer1 = new double*[size];
    for(int i = 0; i < size; ++i)
        buffer1[i] = new double[3];
    for(int i=0; i<imageHeight; i++){
        for(int j=0; j<imageWidth; j++){
            buffer1[i * imageWidth + j][0] = image[i * imageWidth + j][0];
            buffer1[i * imageWidth + j][1] = image[i * imageWidth + j][1];
            buffer1[i * imageWidth + j][2] = image[i * imageWidth + j][2];
        }
    }
    // create second buffer for vertical
    double** buffer2 = new double*[size];
    for(int i = 0; i < size; ++i)
        buffer2[i] = new double[3];
    for(int i=0; i<imageHeight; i++){
        for(int j=0; j<imageWidth; j++){
            buffer2[i * imageWidth + j][0] = image[i * imageWidth + j][0];
            buffer2[i * imageWidth + j][1] = image[i * imageWidth + j][1];
            buffer2[i * imageWidth + j][2] = image[i * imageWidth + j][2];
        }
    }

    // create kernel
    double* kernel = new double[3*3];
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            if(j==2){
                if(i==1) kernel[i*3+j] = 2;
                else kernel[i*3+j] = 1;
            }
            else if(j==1){
                kernel[i*3+j] = 0.0;
            }
            else{
                if(i==1) kernel[i*3+j] = -2;
                else kernel[i*3+j] = -1;
            }
        }
    }

    Convolution(buffer1, kernel, 3, 3, false);

    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            if(i==2){
                if(j==1) kernel[i*3+j] = -2;
                else kernel[i*3+j] = -1;
            }
            else if(i==1){
                kernel[i*3+j] = 0.0;
            }
            else{
                if(j==1) kernel[i*3+j] = 2;
                else kernel[i*3+j] = 1;
            }
        }
    }
    Convolution(buffer2, kernel, 3, 3, false);
    double mag, orien;

    // calculate the magnitude and orientation and store in buffer1 and buffer2
    for(int r=0; r<imageHeight; r++){
        for(int c=0; c<imageWidth; c++){
            mag = sqrt(buffer1[r * imageWidth + c][0]*buffer1[r * imageWidth + c][0] + buffer2[r * imageWidth + c][0]*buffer2[r * imageWidth + c][0]);
            orien = atan2(buffer2[r * imageWidth + c][0], buffer1[r * imageWidth + c][0]);
            buffer1[r * imageWidth + c][0] = mag;
            buffer1[r * imageWidth + c][1] = mag;
            buffer1[r * imageWidth + c][2] = mag;
            buffer2[r * imageWidth + c][0] = orien;
            buffer2[r * imageWidth + c][1] = orien;
            buffer2[r * imageWidth + c][2] = orien;
        }
    }

    // assign (255,255,255) to peak and (0,0,0) otherwise
    for(int r=0; r<imageHeight; r++){
        for(int c=0; c<imageWidth; c++){
            double rgb[3];
            BilinearInterpolation(buffer1, c+cos(buffer2[r * imageWidth + c][0]), r+sin(buffer2[r * imageWidth + c][0]), rgb);
            double rgb2[3];
            BilinearInterpolation(buffer1, c-cos(buffer2[r * imageWidth + c][0]), r-sin(buffer2[r * imageWidth + c][0]), rgb2);
            if(buffer1[r * imageWidth + c][0] > thres && buffer1[r * imageWidth + c][0] >= rgb[0] && buffer1[r * imageWidth + c][0] >= rgb2[0]){
                image[r * imageWidth + c][0] = 255.0;
                image[r * imageWidth + c][1] = 255.0;
                image[r * imageWidth + c][2] = 255.0;
            }
            else{
                image[r * imageWidth + c][0] = 0.0;
                image[r * imageWidth + c][1] = 0.0;
                image[r * imageWidth + c][2] = 0.0;
            }
        }
    }

    // delete resources
    for(int i = 0; i < size; ++i) {
        delete [] buffer1[i];
        delete [] buffer2[i];
    }
    delete [] buffer1;
    delete [] buffer2;
    delete [] kernel;
}

/**************************************************
 TASK 9 (a)
**************************************************/

// Perform K-means clustering on a color image using random seeds
void MainWindow::RandomSeedImage(double** image, int num_clusters)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * num_clusters: number of clusters into which the image is to be clustered
*/
{
    // Add your code here

    // initialize cluster center point
    double **center = new double*[num_clusters];
    for(int i=0; i<num_clusters; i++){
        center[i] = new double[3];
        center[i][0] = rand() % 256;
        center[i][1] = rand() % 256;
        center[i][2] = rand() % 256;
        std::cout<<center[i][0]<<" "<<center[i][1]<<" "<<center[i][2]<<std::endl;
    }

    // calculate the first time cluster
    int *cluster_class = new int[imageWidth*imageHeight];
    int closest_index = 0, dist = 0, min_dist = 1000000, total_dist = 0;
    for(int r=0; r<imageHeight; r++){
        for(int c=0; c<imageWidth; c++){
            for(int i=0; i<num_clusters; i++){
                dist = (abs(image[r*imageWidth + c][0] - center[i][0]) + abs(image[r*imageWidth + c][1] - center[i][1]) + abs(image[r*imageWidth + c][2] - center[i][2]));
                if(dist < min_dist){
                    min_dist = dist;
                    closest_index = i;
                }
            }
            cluster_class[r*imageWidth + c] = closest_index;
            std::cout<<"point "<<r<<" "<<c<<" cluster to "<<closest_index<<std::endl;
            total_dist += min_dist;
            min_dist = 1000000;
        }
    }

    int *cluster_count = new int[num_clusters];
    int last_total_dist = total_dist;
    // max iteration = 100
    for(int i=0; i<99; i++){
        std::cout<<"iter: "<<i<<std::endl;
        // reinitialize center point for each cluster and the counter for each cluster
        for(int j=0; j<num_clusters; j++){
            center[j][0] = 0.0;
            center[j][1] = 0.0;
            center[j][2] = 0.0;
            cluster_count[j] = 0;
        }
        // first find all the new mean point
        for(int r=0; r<imageHeight; r++){
            for(int c=0; c<imageWidth; c++){
                cluster_count[cluster_class[r*imageWidth+c]] ++;
                center[cluster_class[r*imageWidth+c]][0] += image[r*imageWidth+c][0];
                center[cluster_class[r*imageWidth+c]][1] += image[r*imageWidth+c][1];
                center[cluster_class[r*imageWidth+c]][2] += image[r*imageWidth+c][2];
            }
        }
        for(int j=0; j<num_clusters; j++){
            std::cout<<"how many point in cluster "<<j<<": "<<cluster_count[j]<<std::endl;

            // this deal with the edge case that when no pixel is assigned to a cluster
            if(cluster_count[j]!=0){
                center[j][0] /= cluster_count[j];
                center[j][1] /= cluster_count[j];
                center[j][2] /= cluster_count[j];
            }
        }

        // assign cluster
        closest_index = 0; dist = 0; min_dist = 1000000; total_dist = 0;
        for(int r=0; r<imageHeight; r++){
            for(int c=0; c<imageWidth; c++){
                for(int i=0; i<num_clusters; i++){
                    dist = (abs(image[r*imageWidth + c][0] - center[i][0]) + abs(image[r*imageWidth + c][1] - center[i][1]) + abs(image[r*imageWidth + c][2] - center[i][2]));
                    if(dist < min_dist){
                        min_dist = dist;
                        closest_index = i;
                    }
                }
                cluster_class[r*imageWidth + c] = closest_index;
//                std::cout<<"point "<<r<<" "<<c<<" cluster to "<<closest_index<<std::endl;
                total_dist += min_dist;
                min_dist = 1000000;
            }
        }

        // early stop condition
        if(abs(last_total_dist - total_dist) < 30*num_clusters){
            break;
        }
        else{
            last_total_dist = total_dist;
        }
    }

    // make color
    for(int r=0; r<imageHeight; r++){
        for(int c=0; c<imageWidth; c++){
            image[r*imageWidth+c][0] = center[cluster_class[r*imageWidth + c]][0];
            image[r*imageWidth+c][1] = center[cluster_class[r*imageWidth + c]][1];
            image[r*imageWidth+c][2] = center[cluster_class[r*imageWidth + c]][2];
        }
    }

    // print out all the center points for each cluster and delete resources
    for(int i=0; i<num_clusters; i++){
        std::cout<<center[i][0]<<" "<<center[i][1]<<" "<<center[i][2]<<std::endl;
        delete [] center[i];
    }
    delete [] center;
    delete [] cluster_class;
    delete [] cluster_count;
}

/**************************************************
 TASK 9 (b)
**************************************************/

// Perform K-means clustering on a color image using seeds from the image itself
void MainWindow::PixelSeedImage(double** image, int num_clusters)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * num_clusters: number of clusters into which the image is to be clustered
*/
{
    // Add your code here

    // initialize cluster center point
    double **center = new double*[num_clusters];
    for(int i=0; i<num_clusters; i++){
        center[i] = new double[3];
    }

    // first random pick a pixel value
    int random_first = (rand() % imageHeight)*imageWidth + (rand() % imageWidth);
    center[0][0] = image[random_first][0];
    center[0][1] = image[random_first][1];
    center[0][2] = image[random_first][2];

    int compare_index = 0, initial_dist = 0;
    bool flip = true, earlyEnd = false;

    // find other initial center point by comparing the L1 norm with the previous pixel
    // stop when I get num_cluster center points
    for(int r=0; r<imageHeight; r++){
        for(int c=0; c<imageWidth; c++){
            for(int i=0; i<=compare_index; i++){
                initial_dist = (abs(image[r*imageWidth + c][0] - center[i][0]) + abs(image[r*imageWidth + c][1] - center[i][1]) + abs(image[r*imageWidth + c][2] - center[i][2]));
                if(initial_dist < 100){
                    flip = false;
                    break;
                }
            }
            if(flip){
                compare_index ++;
                center[compare_index][0] = image[r*imageWidth + c][0];
                center[compare_index][1] = image[r*imageWidth + c][1];
                center[compare_index][2] = image[r*imageWidth + c][2];
                if(compare_index >= num_clusters-1){
                    earlyEnd = true;
                    break;
                }
            }
            flip = true;
        }
        if(earlyEnd){
            break;
        }
    }
    std::cout<<"create "<<compare_index+1<<" initial center points"<<std::endl;

    // calculate the first time cluster
    int *cluster_class = new int[imageWidth*imageHeight];
    int closest_index = 0, dist = 0, min_dist = 1000000, total_dist = 0;
    for(int r=0; r<imageHeight; r++){
        for(int c=0; c<imageWidth; c++){
            for(int i=0; i<num_clusters; i++){
                dist = (abs(image[r*imageWidth + c][0] - center[i][0]) + abs(image[r*imageWidth + c][1] - center[i][1]) + abs(image[r*imageWidth + c][2] - center[i][2]));
                if(dist < min_dist){
                    min_dist = dist;
                    closest_index = i;
                }
            }
            cluster_class[r*imageWidth + c] = closest_index;
//            std::cout<<"point "<<r<<" "<<c<<" cluster to "<<closest_index<<std::endl;
            total_dist += min_dist;
            min_dist = 1000000;
        }
    }

    int *cluster_count = new int[num_clusters];
    int last_total_dist = total_dist;
    // max iteration = 100
    for(int i=0; i<99; i++){
        std::cout<<"iter: "<<i<<std::endl;
        // reinitialize center point for each cluster and the counter for each cluster
        for(int j=0; j<num_clusters; j++){
            center[j][0] = 0.0;
            center[j][1] = 0.0;
            center[j][2] = 0.0;
            cluster_count[j] = 0;
        }
        // first find all the new mean point
        for(int r=0; r<imageHeight; r++){
            for(int c=0; c<imageWidth; c++){
                cluster_count[cluster_class[r*imageWidth+c]] ++;
                center[cluster_class[r*imageWidth+c]][0] += image[r*imageWidth+c][0];
                center[cluster_class[r*imageWidth+c]][1] += image[r*imageWidth+c][1];
                center[cluster_class[r*imageWidth+c]][2] += image[r*imageWidth+c][2];
            }
        }
        for(int j=0; j<num_clusters; j++){
//            std::cout<<"how many point in cluster "<<j<<": "<<cluster_count[j]<<std::endl;
            if(cluster_count[j]!=0){
                center[j][0] /= cluster_count[j];
                center[j][1] /= cluster_count[j];
                center[j][2] /= cluster_count[j];
            }
        }

        // assign cluster
        closest_index = 0; dist = 0; min_dist = 1000000; total_dist = 0;
        for(int r=0; r<imageHeight; r++){
            for(int c=0; c<imageWidth; c++){
                for(int i=0; i<num_clusters; i++){
                    dist = (abs(image[r*imageWidth + c][0] - center[i][0]) + abs(image[r*imageWidth + c][1] - center[i][1]) + abs(image[r*imageWidth + c][2] - center[i][2]));
                    if(dist < min_dist){
                        min_dist = dist;
                        closest_index = i;
                    }
                }
                cluster_class[r*imageWidth + c] = closest_index;
//                std::cout<<"point "<<r<<" "<<c<<" cluster to "<<closest_index<<std::endl;
                total_dist += min_dist;
                min_dist = 1000000;
            }
        }
        if(abs(last_total_dist - total_dist) < 30*num_clusters){
            break;
        }
        else{
            last_total_dist = total_dist;
        }
    }

    // make color
    for(int r=0; r<imageHeight; r++){
        for(int c=0; c<imageWidth; c++){
            image[r*imageWidth+c][0] = center[cluster_class[r*imageWidth + c]][0];
            image[r*imageWidth+c][1] = center[cluster_class[r*imageWidth + c]][1];
            image[r*imageWidth+c][2] = center[cluster_class[r*imageWidth + c]][2];
        }
    }

    for(int i=0; i<num_clusters; i++){
        std::cout<<center[i][0]<<" "<<center[i][1]<<" "<<center[i][2]<<std::endl;
        delete [] center[i];
    }
    delete [] center;
    delete [] cluster_class;
    delete [] cluster_count;
}


/**************************************************
 EXTRA CREDIT TASKS
**************************************************/

// Perform K-means clustering on a color image using the color histogram
void MainWindow::HistogramSeedImage(double** image, int num_clusters)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * num_clusters: number of clusters into which the image is to be clustered
*/
{
    // Add your code here
}

// Apply the median filter on a noisy image to remove the noise
void MainWindow::MedianImage(double** image, int radius)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * radius: radius of the kernel
*/
{
    // Add your code here

    // create buffer using dynamic 2D array
    int buffer_w = imageWidth + 2*radius;
    int buffer_h = imageHeight + 2*radius;
    int w = imageWidth;
    int h = imageHeight;
    int kernelHeight = 2*radius +1;
    int kernelWidth = 2*radius + 1;
    std::cout<<kernelWidth<<" "<<kernelHeight<<std::endl;
    double** buffer = new double*[buffer_w * buffer_h];
    for(int i = 0; i < buffer_w * buffer_h; ++i)
        buffer[i] = new double[3];

    // assign image to buffer also add zero padding
    for(int i=0; i<buffer_h; i++){
        for(int j=0; j<buffer_w; j++){
            if((i<radius)||(i>=imageHeight+radius)||(j<radius)||(j>=imageWidth+radius)){
                buffer[i * buffer_w + j][0] = 0.0;
                buffer[i * buffer_w + j][1] = 0.0;
                buffer[i * buffer_w + j][2] = 0.0;
            }
            else{
                buffer[i * buffer_w + j][0] = image[(i-radius)*imageWidth + (j-radius)][0];
                buffer[i * buffer_w + j][1] = image[(i-radius)*imageWidth + (j-radius)][1];
                buffer[i * buffer_w + j][2] = image[(i-radius)*imageWidth + (j-radius)][2];
            }
        }
    }

    // ch1 is used to store all the value on image related to the kernel
    std::vector<double> ch1;
    for(int r=kernelHeight/2;r<h+kernelHeight/2;r++)
    {
        for(int c=kernelWidth/2;c<w+kernelWidth/2;c++)
        {
            // store all the value related to kernel into ch1
            for(int rd = -kernelHeight/2; rd <= kernelHeight/2; rd++)
            {
                for(int cd = -kernelWidth/2; cd <= kernelWidth/2; cd++)
                {
//                    std::cout<<rd<<" "<<cd<<std::endl;
                    ch1.push_back(buffer[(r+rd) * buffer_w + (c+cd)][0]);
                }
            }
            std::cout<<ch1.size()<<std::endl; // check I have kernel_width * kernelHeight values in ch1

            // find the median value in ch1 and assign to the pixel
            std::nth_element(ch1.begin(), ch1.begin() + ch1.size()/2, ch1.end());
            image[(r-kernelHeight/2) * w + (c-kernelWidth/2)][0] = ch1[ch1.size()/2];
            image[(r-kernelHeight/2) * w + (c-kernelWidth/2)][1] = ch1[ch1.size()/2];
            image[(r-kernelHeight/2) * w + (c-kernelWidth/2)][2] = ch1[ch1.size()/2];
            ch1.clear();
        }
    }

    // delete resources
    for(int i = 0; i < buffer_w * buffer_h; ++i) {
        delete [] buffer[i];
    }
    delete [] buffer;
}

// Apply Bilater filter on an image
void MainWindow::BilateralImage(double** image, double sigmaS, double sigmaI)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
 * sigmaS: standard deviation in the spatial domain
 * sigmaI: standard deviation in the intensity/range domain
*/
{
    // Add your code here.  Should be similar to GaussianBlurImage.
}

// Perform the Hough transform
void MainWindow::HoughImage(double** image)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
*/
{
    // Add your code here
}

// Perform smart K-means clustering
void MainWindow::SmartKMeans(double** image)
/*
 * image: input image in matrix form of size (imageWidth*imageHeight)*3 having double values
*/
{
    // Add your code here
}
