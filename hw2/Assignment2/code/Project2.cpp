#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include "Matrix.h"
#include <vector>
#include <utility>
#include <iostream>
#include <stdlib.h> 
#include <time.h> 



int imageWidth=0, imageHeight=0;

/*******************************************************************************
    The following are helper routines with code already written.
    The routines you'll need to write for the assignment are below.
*******************************************************************************/

/*******************************************************************************
Blur a single channel floating point image with a Gaussian.
    image - input and output image
    w - image width
    h - image height
    sigma - standard deviation of Gaussian
*******************************************************************************/
void MainWindow::GaussianBlurImage(double *image, int w, int h, double sigma)
{
    int r, c, rd, cd, i;
    int radius = max(1, (int) (sigma*3.0));
    int size = 2*radius + 1;
    double *buffer = new double [w*h];

    memcpy(buffer, image, w*h*sizeof(double));

    if(sigma == 0.0)
        return;

    double *kernel = new double [size];

    for(i=0;i<size;i++)
    {
        double dist = (double) (i - radius);

        kernel[i] = exp(-(dist*dist)/(2.0*sigma*sigma));
    }

    double denom = 0.000001;

    for(i=0;i<size;i++)
        denom += kernel[i];
    for(i=0;i<size;i++)
        kernel[i] /= denom;

    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            double val = 0.0;
            double denom = 0.0;

            for(rd=-radius;rd<=radius;rd++)
                if(r + rd >= 0 && r + rd < h)
                {
                     double weight = kernel[rd + radius];

                     val += weight*buffer[(r + rd)*w + c];
                     denom += weight;
                }

            val /= denom;

            image[r*w + c] = val;
        }
    }

    memcpy(buffer, image, w*h*sizeof(double));

    for(r=0;r<h;r++)
    {
        for(c=0;c<w;c++)
        {
            double val = 0.0;
            double denom = 0.0;

            for(cd=-radius;cd<=radius;cd++)
                if(c + cd >= 0 && c + cd < w)
                {
                     double weight = kernel[cd + radius];

                     val += weight*buffer[r*w + c + cd];
                     denom += weight;
                }

            val /= denom;

            image[r*w + c] = val;
        }
    }


    delete [] kernel;
    delete [] buffer;
}


/*******************************************************************************
Bilinearly interpolate image (helper function for Stitch)
    image - input image
    (x, y) - location to interpolate
    rgb - returned color values
*******************************************************************************/
bool MainWindow::BilinearInterpolation(QImage *image, double x, double y, double rgb[3])
{

    int r = (int) y;
    int c = (int) x;
    double rdel = y - (double) r;
    double cdel = x - (double) c;
    QRgb pixel;
    double del;

    rgb[0] = rgb[1] = rgb[2] = 0.0;

    if(r >= 0 && r < image->height() - 1 && c >= 0 && c < image->width() - 1)
    {
        pixel = image->pixel(c, r);
        del = (1.0 - rdel)*(1.0 - cdel);
        rgb[0] += del*(double) qRed(pixel);
        rgb[1] += del*(double) qGreen(pixel);
        rgb[2] += del*(double) qBlue(pixel);

        pixel = image->pixel(c+1, r);
        del = (1.0 - rdel)*(cdel);
        rgb[0] += del*(double) qRed(pixel);
        rgb[1] += del*(double) qGreen(pixel);
        rgb[2] += del*(double) qBlue(pixel);

        pixel = image->pixel(c, r+1);
        del = (rdel)*(1.0 - cdel);
        rgb[0] += del*(double) qRed(pixel);
        rgb[1] += del*(double) qGreen(pixel);
        rgb[2] += del*(double) qBlue(pixel);

        pixel = image->pixel(c+1, r+1);
        del = (rdel)*(cdel);
        rgb[0] += del*(double) qRed(pixel);
        rgb[1] += del*(double) qGreen(pixel);
        rgb[2] += del*(double) qBlue(pixel);
    }
    else
        return false;

    return true;
}


/*******************************************************************************
Draw detected Harris corners
    cornerPts - corner points
    numCornerPts - number of corner points
    imageDisplay - image used for drawing

    Draws a red cross on top of detected corners
*******************************************************************************/
void MainWindow::DrawCornerPoints(CIntPt *cornerPts, int numCornerPts, QImage &imageDisplay)
{
   int i;
   int r, c, rd, cd;
   int w = imageDisplay.width();
   int h = imageDisplay.height();

   for(i=0;i<numCornerPts;i++)
   {
       c = (int) cornerPts[i].m_X;
       r = (int) cornerPts[i].m_Y;

       for(rd=-2;rd<=2;rd++)
           if(r+rd >= 0 && r+rd < h && c >= 0 && c < w)
               imageDisplay.setPixel(c, r + rd, qRgb(255, 0, 0));

       for(cd=-2;cd<=2;cd++)
           if(r >= 0 && r < h && c + cd >= 0 && c + cd < w)
               imageDisplay.setPixel(c + cd, r, qRgb(255, 0, 0));
   }
}

/*******************************************************************************
Compute corner point descriptors
    image - input image
    cornerPts - array of corner points
    numCornerPts - number of corner points

    If the descriptor cannot be computed, i.e. it's too close to the boundary of
    the image, its descriptor length will be set to 0.

    I've implemented a very simple 8 dimensional descriptor.  Feel free to
    improve upon this.
*******************************************************************************/
void MainWindow::ComputeDescriptors(QImage image, CIntPt *cornerPts, int numCornerPts)
{
    int r, c, cd, rd, i, j;
    int w = image.width();
    int h = image.height();
    double *buffer = new double [w*h];
    QRgb pixel;

    // Descriptor parameters
    double sigma = 2.0;
    int rad = 4;

    // Computer descriptors from green channel
    for(r=0;r<h;r++)
       for(c=0;c<w;c++)
        {
            pixel = image.pixel(c, r);
            buffer[r*w + c] = (double) qGreen(pixel);
        }

    // Blur
    GaussianBlurImage(buffer, w, h, sigma);

    // Compute the desciptor from the difference between the point sampled at its center
    // and eight points sampled around it.
    for(i=0;i<numCornerPts;i++)
    {
        int c = (int) cornerPts[i].m_X;
        int r = (int) cornerPts[i].m_Y;

        if(c >= rad && c < w - rad && r >= rad && r < h - rad)
        {
            double centerValue = buffer[(r)*w + c];
            int j = 0;

            for(rd=-1;rd<=1;rd++)
                for(cd=-1;cd<=1;cd++)
                    if(rd != 0 || cd != 0)
                {
                    cornerPts[i].m_Desc[j] = buffer[(r + rd*rad)*w + c + cd*rad] - centerValue;
                    j++;
                }

            cornerPts[i].m_DescSize = DESC_SIZE;
        }
        else
        {
            cornerPts[i].m_DescSize = 0;
        }
    }

    delete [] buffer;
}

/*******************************************************************************
Draw matches between images
    matches - matching points
    numMatches - number of matching points
    image1Display - image to draw matches
    image2Display - image to draw matches

    Draws a green line between matches
*******************************************************************************/
void MainWindow::DrawMatches(CMatches *matches, int numMatches, QImage &image1Display, QImage &image2Display)
{
    int i;
    // Show matches on image
    QPainter painter;
    painter.begin(&image1Display);
    QColor green(0, 250, 0);
    QColor red(250, 0, 0);

    for(i=0;i<numMatches;i++)
    {
        painter.setPen(green);
        painter.drawLine((int) matches[i].m_X1, (int) matches[i].m_Y1, (int) matches[i].m_X2, (int) matches[i].m_Y2);
        painter.setPen(red);
        painter.drawEllipse((int) matches[i].m_X1-1, (int) matches[i].m_Y1-1, 3, 3);
    }

    QPainter painter2;
    painter2.begin(&image2Display);
    painter2.setPen(green);

    for(i=0;i<numMatches;i++)
    {
        painter2.setPen(green);
        painter2.drawLine((int) matches[i].m_X1, (int) matches[i].m_Y1, (int) matches[i].m_X2, (int) matches[i].m_Y2);
        painter2.setPen(red);
        painter2.drawEllipse((int) matches[i].m_X2-1, (int) matches[i].m_Y2-1, 3, 3);
    }

}


/*******************************************************************************
Given a set of matches computes the "best fitting" homography
    matches - matching points
    numMatches - number of matching points
    h - returned homography
    isForward - direction of the projection (true = image1 -> image2, false = image2 -> image1)
*******************************************************************************/
bool MainWindow::ComputeHomography(CMatches *matches, int numMatches, double h[3][3], bool isForward)
{
    int error;
    int nEq=numMatches*2;

    dmat M=newdmat(0,nEq,0,7,&error);
    dmat a=newdmat(0,7,0,0,&error);
    dmat b=newdmat(0,nEq,0,0,&error);

    double x0, y0, x1, y1;

    for (int i=0;i<nEq/2;i++)
    {
        if(isForward == false)
        {
            x0 = matches[i].m_X1;
            y0 = matches[i].m_Y1;
            x1 = matches[i].m_X2;
            y1 = matches[i].m_Y2;
        }
        else
        {
            x0 = matches[i].m_X2;
            y0 = matches[i].m_Y2;
            x1 = matches[i].m_X1;
            y1 = matches[i].m_Y1;
        }


        //Eq 1 for corrpoint
        M.el[i*2][0]=x1;
        M.el[i*2][1]=y1;
        M.el[i*2][2]=1;
        M.el[i*2][3]=0;
        M.el[i*2][4]=0;
        M.el[i*2][5]=0;
        M.el[i*2][6]=(x1*x0*-1);
        M.el[i*2][7]=(y1*x0*-1);

        b.el[i*2][0]=x0;
        //Eq 2 for corrpoint
        M.el[i*2+1][0]=0;
        M.el[i*2+1][1]=0;
        M.el[i*2+1][2]=0;
        M.el[i*2+1][3]=x1;
        M.el[i*2+1][4]=y1;
        M.el[i*2+1][5]=1;
        M.el[i*2+1][6]=(x1*y0*-1);
        M.el[i*2+1][7]=(y1*y0*-1);

        b.el[i*2+1][0]=y0;

    }
    int ret=solve_system (M,a,b);
    if (ret!=0)
    {
        freemat(M);
        freemat(a);
        freemat(b);

        return false;
    }
    else
    {
        h[0][0]= a.el[0][0];
        h[0][1]= a.el[1][0];
        h[0][2]= a.el[2][0];

        h[1][0]= a.el[3][0];
        h[1][1]= a.el[4][0];
        h[1][2]= a.el[5][0];

        h[2][0]= a.el[6][0];
        h[2][1]= a.el[7][0];
        h[2][2]= 1;
    }

    freemat(M);
    freemat(a);
    freemat(b);

    return true;
}


/*******************************************************************************
*******************************************************************************
*******************************************************************************

    The routines you need to implement are below

*******************************************************************************
*******************************************************************************
*******************************************************************************/
// Convolve the image with the kernel
void Convolution(double* image, double *kernel, int kernelWidth, int kernelHeight, bool add)
{
    // Add your code here

    // create buffer using dynamic 2D array
    int buffer_w = imageWidth + kernelWidth - 1;
    int buffer_h = imageHeight + kernelHeight - 1;
    int w = imageWidth;
    int h = imageHeight;
    double* buffer = new double[buffer_w * buffer_h];

    // if useZeroPdding is set to true, use zero padding, use fixed padding otherwise
    bool useZeroPdding = false;
    if(useZeroPdding){
        for(int i=0; i<buffer_h; i++){
            for(int j=0; j<buffer_w; j++){
                if((i<kernelHeight/2)||(i>=imageHeight+kernelHeight/2)||(j<kernelWidth/2)||(j>=imageWidth+kernelWidth/2)){
                    buffer[i * buffer_w + j] = 0.0;
                }
                else{
                    buffer[i * buffer_w + j] = image[(i-kernelHeight/2)*imageWidth + (j-kernelWidth/2)];
                }
            }
        }
    }
    else{
        for(int i=0; i<buffer_h; i++){
            for(int j=0; j<buffer_w; j++){
                if((i<kernelHeight/2) && (j<kernelWidth/2)){
                    buffer[i * buffer_w + j] = image[0];
                }
                else if((i<kernelHeight/2) && (j>=imageWidth+kernelWidth/2)){
                    buffer[i * buffer_w + j] = image[imageWidth-1];
                }
                else if((j<kernelWidth/2) && (i>=imageHeight+kernelHeight/2)){
                    buffer[i * buffer_w + j] = image[(imageHeight-1)*imageWidth];
                }
                else if((j>=imageWidth+kernelWidth/2) && (i>=imageHeight+kernelHeight/2)){
                    buffer[i * buffer_w + j] = image[imageHeight*imageWidth-1];
                }
                else if((i<kernelHeight/2)){
                    buffer[i * buffer_w + j] = image[0*imageWidth + (j-kernelWidth/2)];
                }
                else if((j<kernelWidth/2)){
                    buffer[i * buffer_w + j] = image[(i-kernelHeight/2)*imageWidth + 0];
                }
                else if((i>=imageHeight+kernelHeight/2)){
                    buffer[i * buffer_w + j] = image[(imageHeight-1)*imageWidth + (j-kernelWidth/2)];
                }
                else if((j>=imageWidth+kernelWidth/2)){
                    buffer[i * buffer_w + j] = image[(i-kernelHeight/2)*imageWidth + (imageWidth-1)];
                }
                else{
                    buffer[i * buffer_w + j] = image[(i-kernelHeight/2)*imageWidth + (j-kernelWidth/2)];
                }
            }
        }
    }
    // do the convolution
    for(int r=kernelHeight/2;r<h+kernelHeight/2;r++)
    {
        for(int c=kernelWidth/2;c<w+kernelWidth/2;c++)
        {
            double rgb;
            rgb = 0.0;

            // Convolve the kernel at each pixel
            for(int rd = -kernelHeight/2; rd <= kernelHeight/2; rd++)
            {
                for(int cd = -kernelWidth/2; cd <= kernelWidth/2; cd++)
                {
                     double pixel = buffer[(r+rd) * buffer_w + (c+cd)];

                     // Get the value of the kernel
                     double weight = kernel[(rd + (kernelHeight/2))*kernelWidth + cd + (kernelWidth/2)];

                     rgb += weight*(double) pixel;
                }
            }
            // Store the pixel in the image to be returned
            // You need to store the RGB values in the double form of the image
            if(!add)
            {
                image[(r-kernelHeight/2) * w + (c-kernelWidth/2)] = rgb;
            }
            else
            {
                image[(r-kernelHeight/2) * w + (c-kernelWidth/2)] = rgb+128;
            }
        }
    }

    // delete dynamic allocated memory
    delete [] buffer;
}

// Compute the First derivative of an image along the horizontal direction and then apply Gaussian blur.
void FirstDerivImage_x(double* image, double sigma)
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
    Convolution(image, kernel, 3, 3, false);
    // GaussianBlurImage(image, imageWidth, imageHeight, sigma);
    // Clean up
    delete[] kernel;
}

// Compute the First derivative of an image along the vertical direction and then apply Gaussian blur.
void FirstDerivImage_y(double* image, double sigma)
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
    Convolution(image, kernel, 3, 3, false);
    // GaussianBlurImage(image, imageWidth, imageHeight, sigma);
    // Clean up
    delete[] kernel;
}

/*******************************************************************************
Detect Harris corners.
    image - input image
    sigma - standard deviation of Gaussian used to blur corner detector
    thres - Threshold for detecting corners
    cornerPts - returned corner points
    numCornerPts - number of corner points returned
    imageDisplay - image returned to display (for debugging)
*******************************************************************************/
void MainWindow::HarrisCornerDetector(QImage image, double sigma, double thres, CIntPt **cornerPts, int &numCornerPts, QImage &imageDisplay)
{
    int r, c;
    int w = image.width();
    int h = image.height();
    double *buffer = new double [w*h];
    QRgb pixel;

    numCornerPts = 0;

    // Compute the corner response using just the green channel
    for(r=0;r<h;r++)
    {
       for(c=0;c<w;c++)
        {
            pixel = image.pixel(c, r);

            buffer[r*w + c] = (double) qGreen(pixel);
        }
    }

    // Write your Harris corner detection code here.

    // create first buffer for vertical
    imageWidth = image.width();
    imageHeight = image.height();
    std:: cout<<imageWidth << " "<<imageHeight<<std::endl;
    int size = imageWidth * imageHeight;
    double* buffer1 = new double[size];
    double* buffer2 = new double[size];
    double* buffer3 = new double[size];

    // GaussianBlurImage(buffer, imageWidth, imageHeight, sigma);
    for(int i=0; i<imageHeight; i++){
        for(int j=0; j<imageWidth; j++){
            buffer1[i * imageWidth + j] = buffer[i * imageWidth + j];
            buffer2[i * imageWidth + j] = buffer[i * imageWidth + j];
        }
    }

    // i step get Ix^2 Iy^2 IxIy
    FirstDerivImage_x(buffer1, sigma);
    FirstDerivImage_y(buffer2, sigma);

    for(int i=0; i<imageHeight; i++){
        for(int j=0; j<imageWidth; j++){
            buffer3[i * imageWidth + j] = buffer1[i * imageWidth + j] * buffer2[i * imageWidth + j];
            buffer1[i * imageWidth + j] = buffer1[i * imageWidth + j] * buffer1[i * imageWidth + j];
            buffer2[i * imageWidth + j] = buffer2[i * imageWidth + j] * buffer2[i * imageWidth + j];
        }
    }
    GaussianBlurImage(buffer1, imageWidth, imageHeight, sigma);
    GaussianBlurImage(buffer2, imageWidth, imageHeight, sigma);
    GaussianBlurImage(buffer3, imageWidth, imageHeight, sigma);

    // for debug
    // for(int i=0; i<h; i++){
    //     for(int j=0; j<w; j++){
    //         std::cout<<buffer1[i * imageWidth + j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }

    int radius = 2;
    int buffer_w = imageWidth + 2*radius;
    int buffer_h = imageHeight + 2*radius;
    int kernelHeight = 2*radius +1;
    int kernelWidth = 2*radius + 1;
    double* R = new double[buffer_w * buffer_h];
    double aa, bb, cc, dd;
    for(int i=0; i<buffer_h; i++){
        for(int j=0; j<buffer_w; j++){
            if((i<radius)||(i>=imageHeight+radius)||(j<radius)||(j>=imageWidth+radius)){
                R[i * buffer_w + j] = 0.0;
            }
            else{
                aa = buffer1[(i-kernelHeight/2)*imageWidth + (j-kernelWidth/2)];
                bb = buffer3[(i-kernelHeight/2)*imageWidth + (j-kernelWidth/2)];
                cc = buffer3[(i-kernelHeight/2)*imageWidth + (j-kernelWidth/2)];
                dd = buffer2[(i-kernelHeight/2)*imageWidth + (j-kernelWidth/2)];
                // std::cout<<aa<<" "<<bb<<" "<<cc<<" "<<dd<<" "<<((aa*dd - bb*cc)/(aa+dd))<<std::endl;
                if(aa+dd!=0)
                    R[i * buffer_w + j] = ((aa*dd - bb*cc)/(aa+dd)) > thres ? (aa*dd - bb*cc)/(aa+dd) : 0.0;
                else
                    R[i * buffer_w + j] = 0.0;
            }
        }
    }

    // for debug
    // for(int i=0; i<buffer_h; i++){
    //     for(int j=0; j<buffer_w; j++){
    //         std::cout<<R[i * w + j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }

    bool flip = false;
    std::vector<std::pair<int, int> > corner;
    for(int r=kernelHeight/2;r<h+kernelHeight/2;r++)
    {
        for(int c=kernelWidth/2;c<w+kernelWidth/2;c++)
        {
            // std::cout<<r-kernelHeight/2<< " "<< c-kernelWidth/2<<std::endl;
            for(int rd = -kernelHeight/2; rd <= kernelHeight/2; rd++)
            {
                for(int cd = -kernelWidth/2; cd <= kernelWidth/2; cd++)
                {
                    if(R[(r+rd) * buffer_w + (c+cd)] > R[(r+0) * buffer_w + (c+0)])
                    {   
                        flip = true;
                        break;
                    }
                }
                if(flip) break;
            }
            if(!flip && R[(r+0) * buffer_w + (c+0)]!=0.0)
            {
                corner.push_back({r-kernelHeight/2, c-kernelWidth/2});
            }
            flip = false;
            
        }
    }
    // std::cout<<"finish"<<std::endl;

    numCornerPts = corner.size();
    *cornerPts = new CIntPt [numCornerPts];
    for(int i=0; i<numCornerPts; i++){
        (*cornerPts)[i].m_X = corner[i].second;
        (*cornerPts)[i].m_Y = corner[i].first;
        // std::cout<<corner[i].second<<" "<<corner[i].first<<std::endl;
    }



    // Once you know the number of corner points allocate an array as follows:
    // *cornerPts = new CIntPt [numCornerPts];
    // Access the values using: (*cornerPts)[i].m_X = 5.0;
    //
    // The position of the corner point is (m_X, m_Y)
    // The descriptor of the corner point is stored in m_Desc
    // The length of the descriptor is m_DescSize, if m_DescSize = 0, then it is not valid.

    // Once you are done finding the corner points, display them on the image
    DrawCornerPoints(*cornerPts, numCornerPts, imageDisplay);

    delete [] buffer;
    delete [] buffer1;
    delete [] buffer2;
    delete [] buffer3;
    delete [] R;
}


/*******************************************************************************
Find matching corner points between images.
    image1 - first input image
    cornerPts1 - corner points corresponding to image 1
    numCornerPts1 - number of corner points in image 1
    image2 - second input image
    cornerPts2 - corner points corresponding to image 2
    numCornerPts2 - number of corner points in image 2
    matches - set of matching points to be returned
    numMatches - number of matching points returned
    image1Display - image used to display matches
    image2Display - image used to display matches
*******************************************************************************/
void MainWindow::MatchCornerPoints(QImage image1, CIntPt *cornerPts1, int numCornerPts1,
                             QImage image2, CIntPt *cornerPts2, int numCornerPts2,
                             CMatches **matches, int &numMatches, QImage &image1Display, QImage &image2Display)
{
    numMatches = 0;

    // Compute the descriptors for each corner point.
    // You can access the descriptor for each corner point using cornerPts1[i].m_Desc[j].
    // If cornerPts1[i].m_DescSize = 0, it was not able to compute a descriptor for that point
    ComputeDescriptors(image1, cornerPts1, numCornerPts1);
    ComputeDescriptors(image2, cornerPts2, numCornerPts2);

    // Add your code here for finding the best matches for each point.
    double min_dist = 1000000, dist = 0;
    std::pair<int, int> min_pair;
    std::vector<std::pair<int, int> > match_pair;
    for(int i = 0; i < numCornerPts1; i++){
        if(cornerPts1[i].m_DescSize != 0){
            for(int j = 0; j < numCornerPts2; j++){
                if(cornerPts2[j].m_DescSize != 0){
                    for(int k=0; k<cornerPts1[i].m_DescSize; k++){
                        dist += fabs(cornerPts1[i].m_Desc[k] - cornerPts2[j].m_Desc[k]);
                    }
                    if(dist < min_dist){
                        min_dist = dist;
                        min_pair = {i, j};
                    }
                    dist = 0;
                }
            }
            match_pair.push_back(min_pair);
            min_dist = 1000000;

        }
    }

    numMatches = match_pair.size();
    *matches = new CMatches [numMatches];
    for(int i=0; i<numMatches; i++){
        (*matches)[i].m_X1 = cornerPts1[match_pair[i].first].m_X;
        (*matches)[i].m_Y1 = cornerPts1[match_pair[i].first].m_Y;
        (*matches)[i].m_X2 = cornerPts2[match_pair[i].second].m_X;
        (*matches)[i].m_Y2 = cornerPts2[match_pair[i].second].m_Y;
    }

    // Once you uknow the number of matches allocate an array as follows:
    // *matches = new CMatches [numMatches];
    //
    // The position of the corner point in iamge 1 is (m_X1, m_Y1)
    // The position of the corner point in image 2 is (m_X2, m_Y2)

    // Draw the matches
    DrawMatches(*matches, numMatches, image1Display, image2Display);
}

/*******************************************************************************
Project a point (x1, y1) using the homography transformation h
    (x1, y1) - input point
    (x2, y2) - returned point
    h - input homography used to project point
*******************************************************************************/
void MainWindow::Project(double x1, double y1, double &x2, double &y2, double h[3][3])
{
    // Add your code here.
    x2 = (h[0][0]*x1 + h[0][1]*y1 + h[0][2]*1)/(h[2][0]*x1 + h[2][1]*y1 + h[2][2]*1);
    y2 = (h[1][0]*x1 + h[1][1]*y1 + h[1][2]*1)/(h[2][0]*x1 + h[2][1]*y1 + h[2][2]*1);
}

/*******************************************************************************
Count the number of inliers given a homography.  This is a helper function for RANSAC.
    h - input homography used to project points (image1 -> image2
    matches - array of matching points
    numMatches - number of matchs in the array
    inlierThreshold - maximum distance between points that are considered to be inliers

    Returns the total number of inliers.
*******************************************************************************/
int MainWindow::ComputeInlierCount(double h[3][3], CMatches *matches, int numMatches, double inlierThreshold)
{
    // Add your code here.
    double proj_x, proj_y;
    int inlier_count = 0;
    for(int i = 0; i < numMatches; i++){
        Project(matches[i].m_X1, matches[i].m_Y1, proj_x, proj_y, h);
        if(sqrt((matches[i].m_X2 - proj_x)*(matches[i].m_X2 - proj_x) + (matches[i].m_Y2 - proj_y)*(matches[i].m_Y2 - proj_y)) < inlierThreshold){
            inlier_count ++;
        }
    }
    return inlier_count;
}


/*******************************************************************************
Compute homography transformation between images using RANSAC.
    matches - set of matching points between images
    numMatches - number of matching points
    numIterations - number of iterations to run RANSAC
    inlierThreshold - maximum distance between points that are considered to be inliers
    hom - returned homography transformation (image1 -> image2)
    homInv - returned inverse homography transformation (image2 -> image1)
    image1Display - image used to display matches
    image2Display - image used to display matches
*******************************************************************************/
void MainWindow::RANSAC(CMatches *matches, int numMatches, int numIterations, double inlierThreshold,
                        double hom[3][3], double homInv[3][3], QImage &image1Display, QImage &image2Display)
{
    // Add your code here.
    srand (time(NULL));
    int n1, n2, n3, n4;
    CMatches temp_match[4];
    int max_inlier_count = 0, inlier_count;
    double temp_hom[3][3];

    for(int i=0; i<numIterations; i++){
        n1 = rand() % numMatches + 1;
        n2 = rand() % numMatches + 1;
        n3 = rand() % numMatches + 1;
        n4 = rand() % numMatches + 1;
        temp_match[0] = matches[n1];
        temp_match[1] = matches[n2];
        temp_match[2] = matches[n3];
        temp_match[3] = matches[n4];
        ComputeHomography(temp_match, 4, temp_hom, true);
        inlier_count = ComputeInlierCount(temp_hom, matches, numMatches, inlierThreshold);
        if(inlier_count > max_inlier_count){
            max_inlier_count = inlier_count;
            hom[0][0] = temp_hom[0][0]; hom[0][1] = temp_hom[0][1]; hom[0][2] = temp_hom[0][2]; 
            hom[1][0] = temp_hom[1][0]; hom[1][1] = temp_hom[1][1]; hom[1][2] = temp_hom[1][2]; 
            hom[2][0] = temp_hom[2][0]; hom[2][1] = temp_hom[2][1]; hom[2][2] = temp_hom[2][2]; 
        }
    }

    std::vector<CMatches> final_match_vec;
    double proj_x, proj_y;
    for(int i = 0; i < numMatches; i++){
        Project(matches[i].m_X1, matches[i].m_Y1, proj_x, proj_y, hom);
        if(sqrt((matches[i].m_X2 - proj_x)*(matches[i].m_X2 - proj_x) + (matches[i].m_Y2 - proj_y)*(matches[i].m_Y2 - proj_y)) < inlierThreshold){
            final_match_vec.push_back(matches[i]);
        }
    }

    CMatches *final_match = new CMatches[final_match_vec.size()];
    for(int i = 0; i < final_match_vec.size(); i++){
        final_match[i] = final_match_vec[i];
    }
    ComputeHomography(final_match, final_match_vec.size(), hom, true);
    ComputeHomography(final_match, final_match_vec.size(), homInv, false);

    DrawMatches(final_match, final_match_vec.size(), image1Display, image2Display);
    // After you're done computing the inliers, display the corresponding matches.
    //DrawMatches(inliers, numInliers, image1Display, image2Display);

}

/*******************************************************************************
Stitch together two images using the homography transformation
    image1 - first input image
    image2 - second input image
    hom - homography transformation (image1 -> image2)
    homInv - inverse homography transformation (image2 -> image1)
    stitchedImage - returned stitched image
*******************************************************************************/
void MainWindow::Stitch(QImage image1, QImage image2, double hom[3][3], double homInv[3][3], QImage &stitchedImage)
{
    // Width and height of stitchedImage
    int ws = 0;
    int hs = 0;

    // Add your code to compute ws and hs here.
    double proj_1x, proj_1y, proj_2x, proj_2y, proj_3x, proj_3y, proj_4x, proj_4y;
    Project(0.0, 0.0, proj_1x, proj_1y, homInv);
    Project(image2.width(), 0.0, proj_2x, proj_2y, homInv);
    Project(0.0, image2.height(), proj_3x, proj_3y, homInv);
    Project(image2.width(), image2.height(), proj_4x, proj_4y, homInv);

    // std::cout<<proj_1x<< " "<<proj_1y<<std::endl;
    // std::cout<<proj_2x<< " "<<proj_2y<<std::endl;
    // std::cout<<proj_3x<< " "<<proj_3y<<std::endl;
    // std::cout<<proj_4x<< " "<<proj_4y<<std::endl;
    double ub, db, lb, rb;
    rb = max((double)image1.width(), max(proj_1x, max(proj_2x, max(proj_3x, proj_4x))));
    ub = max((double)image1.height(), max(proj_1y, max(proj_2y, max(proj_3y, proj_4y))));
    lb = min(0.0, min(proj_1x, min(proj_2x, min(proj_3x, proj_4x))));
    db = min(0.0, min(proj_1y, min(proj_2y, min(proj_3y, proj_4y))));

    ws = rb - lb;
    hs = ub - db;

    stitchedImage = QImage(ws, hs, QImage::Format_RGB32);
    stitchedImage.fill(qRgb(0,0,0));

    // Add you code to warp image1 and image2 to stitchedImage here.

    for(int i=0; i<image1.height(); i++){
        for(int j=0; j<image1.width(); j++){
            stitchedImage.setPixel(j-lb, i-db, image1.pixel(j, i));
        }
    }

    double proj_x, proj_y;
    bool success;
    double rgb[3];
    for(int i=db; i<ub; i++){
        for(int j=lb; j<rb; j++){
            Project(j, i, proj_x, proj_y, hom);
            if(proj_x > 0 && proj_x < image2.width() && proj_y > 0 && proj_y < image2.height()){
                success = BilinearInterpolation(&image2, proj_x, proj_y, rgb);
                stitchedImage.setPixel(j-lb, i-db, qRgb(rgb[0], rgb[1], rgb[2]));
            }
        }
    }
}

