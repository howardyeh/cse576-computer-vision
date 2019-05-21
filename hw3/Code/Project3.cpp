#include "mainwindow.h"
#include "math.h"
#include "ui_mainwindow.h"
#include <QtGui>
#include "stdlib.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <utility>


/**************************************************
CODE FOR K-MEANS COLOR IMAGE CLUSTERING (RANDOM SEED)
**************************************************/

void Clustering(QImage *image, int num_clusters, int maxit)
{
        int w = image->width(), h = image->height();
        QImage buffer = image->copy();

        std::vector<QRgb> centers, centers_new;

        //initialize random centers
        int n = 1;
        while (n <= num_clusters)
        {
            QRgb center = qRgb(rand() % 256, rand() % 256, rand() % 256);
            centers.push_back(center);
            centers_new.push_back(center);
            n++;
        }

        //iterative part
        int it = 0;
        std::vector<int> ids;
        while (it < maxit)
        {
                ids.clear();
                //assign pixels to clusters
                for (int r = 0; r < h; r++)
                	for (int c = 0; c < w; c++)
                	{
                        int maxd = 999999, id = 0;
                        for (int n = 0; n < num_clusters; n++)
                        {
                                QRgb pcenter = centers[n];
                                QRgb pnow = buffer.pixel(c, r);
                                int d = abs(qRed(pcenter) - qRed(pnow)) + abs(qGreen(pcenter) - qGreen(pnow)) + abs(qBlue(pcenter) - qBlue(pnow));
                                if (d < maxd)
                                {
                                        maxd = d; id = n;
                                }
                        }
                        ids.push_back(id);
                	}

                //update centers
                std::vector<int> cnt, rs, gs, bs;
                for (int n = 0; n < num_clusters; n++)
                {
                        rs.push_back(0); gs.push_back(0); bs.push_back(0); cnt.push_back(0);
                }
                for (int r = 0; r < h; r++)
                    for (int c = 0; c < w; c++)
                    {
                        QRgb pixel = buffer.pixel(c,r);
                        rs[ids[r * w + c]] += qRed(pixel);
                        gs[ids[r * w + c]] += qGreen(pixel);
                        bs[ids[r * w + c]] += qBlue(pixel);
                        cnt[ids[r * w + c]]++;
                    }
                for (int n = 0; n < num_clusters; n++)
                    if (cnt[n] == 0) // no pixels in a cluster
                        continue;
                    else
                        centers_new[n] = qRgb(rs[n]/cnt[n], gs[n]/cnt[n], bs[n]/cnt[n]);

                centers = centers_new; it++;
        }
        //render results
        for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++)
                image->setPixel(c, r, qRgb(ids[r * w + c],ids[r * w + c],ids[r * w + c]));
}

/**************************************************
CODE FOR FINDING CONNECTED COMPONENTS
**************************************************/

#include "utils.h"

#define MAX_LABELS 80000

#define I(x,y)   (image[(y)*(width)+(x)])
#define N(x,y)   (nimage[(y)*(width)+(x)])

void uf_union( int x, int y, unsigned int parent[] )
{
    while ( parent[x] )
        x = parent[x];
    while ( parent[y] )
        y = parent[y];
    if ( x != y ) {
        if ( y < x ) parent[x] = y;
        else parent[y] = x;
    }
}

int next_label = 1;

int uf_find( int x, unsigned int parent[], unsigned int label[] )
{
    while ( parent[x] )
        x = parent[x];
    if ( label[x] == 0 )
        label[x] = next_label++;
    return label[x];
}

void conrgn(int *image, int *nimage, int width, int height)
{
    unsigned int parent[MAX_LABELS], labels[MAX_LABELS];
    int next_region = 1, k;

    memset( parent, 0, sizeof(parent) );
    memset( labels, 0, sizeof(labels) );

    for ( int y = 0; y < height; ++y )
    {
        for ( int x = 0; x < width; ++x )
        {
            k = 0;
            if ( x > 0 && I(x-1,y) == I(x,y) )
                k = N(x-1,y);
            if ( y > 0 && I(x,y-1) == I(x,y) && N(x,y-1) < k )
                k = N(x,y-1);
            if ( k == 0 )
            {
                k = next_region; next_region++;
            }
            if ( k >= MAX_LABELS )
            {
                fprintf(stderr, "Maximum number of labels reached. Increase MAX_LABELS and recompile.\n"); exit(1);
            }
            N(x,y) = k;
            if ( x > 0 && I(x-1,y) == I(x,y) && N(x-1,y) != k )
                uf_union( k, N(x-1,y), parent );
            if ( y > 0 && I(x,y-1) == I(x,y) && N(x,y-1) != k )
                uf_union( k, N(x,y-1), parent );
        }
    }
    for ( int i = 0; i < width*height; ++i )
        if ( nimage[i] != 0 )
            nimage[i] = uf_find( nimage[i], parent, labels );

    next_label = 1; // reset its value to its initial value
    return;
}


/**************************************************
 **************************************************
TIME TO WRITE CODE
**************************************************
**************************************************/


/**************************************************
Code to compute the features of a given image (both database images and query image)
**************************************************/

std::vector<double*> MainWindow::ExtractFeatureVector(QImage image)
{
    /********** STEP 1 **********/

    // Display the start of execution of this step in the progress box of the application window
    // You can use these 2 lines to display anything you want at any point of time while debugging

    ui->progressBox->append(QString::fromStdString("Clustering.."));
    QApplication::processEvents();

    // Perform K-means color clustering
    // This time the algorithm returns the cluster id for each pixel, not the rgb values of the corresponding cluster center
    // The code for random seed clustering is provided. You are free to use any clustering algorithm of your choice from HW 1
    // Experiment with the num_clusters and max_iterations values to get the best result

    int num_clusters = 5;
    int max_iterations = 50;
    QImage image_copy = image;
    Clustering(&image_copy,num_clusters,max_iterations);


    /********** STEP 2 **********/


    ui->progressBox->append(QString::fromStdString("Connecting components.."));
    QApplication::processEvents();

    // Find connected components in the labeled segmented image
    // Code is given, you don't need to change

    int r, c, w = image_copy.width(), h = image_copy.height();
    int *img = (int*)malloc(w*h * sizeof(int));
    memset( img, 0, w * h * sizeof( int ) );
    int *nimg = (int*)malloc(w*h *sizeof(int));
    memset( nimg, 0, w * h * sizeof( int ) );

    for (r=0; r<h; r++)
        for (c=0; c<w; c++)
            img[r*w + c] = qRed(image_copy.pixel(c,r));

    conrgn(img, nimg, w, h);

    int num_regions=0;
    for (r=0; r<h; r++)
        for (c=0; c<w; c++)
            num_regions = (nimg[r*w+c]>num_regions)? nimg[r*w+c]: num_regions;

    ui->progressBox->append(QString::fromStdString("#regions = "+std::to_string(num_regions)));
    QApplication::processEvents();

    // The resultant image of Step 2 is 'nimg', whose values range from 1 to num_regions

    // WRITE YOUR REGION THRESHOLDING AND REFINEMENT CODE HERE
    
    // first count the number of pixel in each region
    int *count_num_in_region = new int[num_regions];
    memset( count_num_in_region, 0, num_regions * sizeof( int ) );
    for(int i=0; i<w*h; i++){
        count_num_in_region[nimg[i]-1] ++;
    }

    // record the mapping for region that pass the threshold
    int num_in_region_thres = 200;
    int num_region_after_thres = 0;
    std::map<int, int> region_after_thres;
    for(int i=0; i<num_regions; i++){
        if(count_num_in_region[i] > num_in_region_thres){
            if(region_after_thres.find(i+1)==region_after_thres.end()){
                num_region_after_thres ++;
                region_after_thres[i+1] = num_region_after_thres;
            }
        }
    }
    delete [] count_num_in_region;
    ui->progressBox->append(QString::fromStdString("#regions after thres = "+std::to_string(num_region_after_thres)));
    QApplication::processEvents();

    // ignore the region that do not pass the threshold, and map the region that pass the threshold
    num_regions = num_region_after_thres;
    for(r=0; r<h; r++){
        for(c=0; c<w; c++){
            if(region_after_thres.find(nimg[r*w+c])==region_after_thres.end()){
                nimg[r*w+c] = 0;
            }
            else{
                nimg[r*w+c] = region_after_thres[nimg[r*w+c]];
            }
        }
    }


    /********** STEP 3 **********/


    ui->progressBox->append(QString::fromStdString("Extracting features.."));
    QApplication::processEvents();

    // Extract the feature vector of each region

    // Set length of feature vector according to the number of features you plan to use.
    featurevectorlength = 13;

    // Initializations required to compute feature vector

    std::vector<double*> featurevector; // final feature vector of the image; to be returned
    double **features = new double* [num_regions]; // stores the feature vector for each connected component
    for(int i=0;i<num_regions; i++)
        features[i] = new double[featurevectorlength](); // initialize with zeros

    // Sample code for computing the mean RGB values and size of each connected component

    for(int r=0; r<h; r++){
        for (int c=0; c<w; c++)
        {
            // because 0 is set to ignore, so need to add if nimg[r*w+c] != 0
            if(nimg[r*w+c] != 0){
                features[nimg[r*w+c]-1][0] += 1; // stores the number of pixels for each connected component
                features[nimg[r*w+c]-1][1] += (double) qRed(image.pixel(c,r));
                features[nimg[r*w+c]-1][2] += (double) qGreen(image.pixel(c,r));
                features[nimg[r*w+c]-1][3] += (double) qBlue(image.pixel(c,r));
            }
        }
    }

    // calculate gray-level cooccurrence matrix
    QImage gray = image.convertToFormat(QImage::Format_Grayscale8);
    map<int, map<pair<int, int>, int> > glcm_for_each_region;
    map<int, double> glcm_for_each_region_count;
    for(int r=0; r<h-1; r++){
        for(int c=0; c<w-1; c++){
            if(nimg[r*w+c] != 0 && nimg[r*w+c] == nimg[(r+1)*w+c+1]){
                glcm_for_each_region[nimg[r*w+c]][{(int) qRed(gray.pixel(c,r)), (int) qRed(gray.pixel(c+1,r+1))}] ++;
                glcm_for_each_region_count[nimg[r*w+c]] ++;
            }
        }
    }

    // --- for debugging ---
    // for(auto i:glcm_for_each_region){
    //     std::cout<<i.first<<std::endl;
    // }
    // for(auto i:glcm_for_each_region[1]){
    //     std::cout<<i.first.first<<" "<<i.first.second<<" "<<i.second<<std::endl;
    // }
    // std::cout<<glcm_for_each_region_count[1]<<std::endl;
    // ----------------------

    // calculate energy, entropy, and contrast
    double energy = 0.0;
    double entropy = 0.0;
    double contrast = 0.0;
    for(auto i : glcm_for_each_region){
        energy = 0.0; entropy = 0.0; contrast = 0.0;
        for(auto j : i.second){
            energy += (j.second/glcm_for_each_region_count[i.first])*(j.second/glcm_for_each_region_count[i.first]);
            entropy += (j.second/glcm_for_each_region_count[i.first])*log2((j.second/glcm_for_each_region_count[i.first]));
            contrast += (j.first.first - j.first.second)*(j.first.first - j.first.second)*(j.second/glcm_for_each_region_count[i.first]);
        }
        features[i.first-1][4] = energy;
        features[i.first-1][5] = entropy;
        features[i.first-1][6] = contrast;
        // std::cout<<"energy of region "<<i.first<<" = "<<energy<<std::endl;
    }


    // calculate centroid mean
    map<int, pair<double, double> > centroid_of_region;
    map<int, double> count_centroid_of_region;
    for(int r=0; r<h; r++){
        for (int c=0; c<w; c++)
        {
            // because 0 is set to ignore, so need to add if nimg[r*w+c] != 0
            if(nimg[r*w+c] != 0){
                centroid_of_region[nimg[r*w+c]].first += r;
                centroid_of_region[nimg[r*w+c]].second += c;
                count_centroid_of_region[nimg[r*w+c]] ++;
            }
        }
    }
    for(auto i : centroid_of_region){
        features[i.first-1][7] = i.second.first/count_centroid_of_region[i.first];
        features[i.first-1][8] = i.second.second/count_centroid_of_region[i.first];
        // features[i.first-1][9] = count_centroid_of_region[i.first];
    }

    // calculate bounding box
    map<int, vector<int> > boundary;
    for(int r=0; r<h; r++){
        for (int c=0; c<w; c++)
        {
            // because 0 is set to ignore, so need to add if nimg[r*w+c] != 0
            if(nimg[r*w+c] != 0){
                if(boundary.find(nimg[r*w+c])==boundary.end()){
                    boundary[nimg[r*w+c]] = {r, r, c, c};
                }
                else{
                    if(boundary[nimg[r*w+c]][0] > r) boundary[nimg[r*w+c]][0] = r;
                    if(boundary[nimg[r*w+c]][1] < r) boundary[nimg[r*w+c]][1] = r;
                    if(boundary[nimg[r*w+c]][2] > c) boundary[nimg[r*w+c]][2] = c;
                    if(boundary[nimg[r*w+c]][3] < c) boundary[nimg[r*w+c]][3] = c;
                }
            }
        }
    }

    for(auto i : boundary){
        features[i.first-1][9] = i.second[0]+1;
        features[i.first-1][10] = i.second[1]+1;
        features[i.first-1][11] = i.second[2]+1;
        features[i.first-1][12] = i.second[3]+1;
    }
    

    // normalize using min max 
    double temp_max=-100000, temp_min=100000;
    for(int n=4; n<featurevectorlength; n++){
        temp_max=-100000, temp_min=100000;
        for(int m=0; m<num_regions; m++){
            temp_max = std::max(features[m][n], temp_max);
            temp_min = std::min(features[m][n], temp_min);
        }
        for(int m=0; m<num_regions; m++){
            features[m][n] = (features[m][n] - temp_min) / (temp_max - temp_min);
        }
    }

    // Save the mean RGB and size values as image feature after normalization
    for(int m=0; m<num_regions; m++)
    {
        for(int n=1; n<4; n++)
            features[m][n] /= features[m][0]*255.0;
        features[m][0] /= (double) w*h;
        featurevector.push_back(features[m]);
    }

    

    // Return the created feature vector
    ui->progressBox->append(QString::fromStdString("***Done***"));
    QApplication::processEvents();
    return featurevector;
}


/***** Code to compute the distance between two images *****/

// Function that implements distance measure 1
double distance1(double* vector1, double* vector2, int length)
{
    // use L2 norm
    // std::cout<<"hi"<<std::endl;
    double w[13] = {1.0, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2};
    double d = 0.0;
    for(int i=0; i<length; i++){
        // std::cout<<i<<" "<<vector1[i]<<" "<<vector2[i]<<std::endl;
        d += w[i]*((vector1[i] - vector2[i])*(vector1[i] - vector2[i]));
    }
    return d;
}


double distance2(double* vector1, double* vector2, int length)
{
    // use cross entropy
    double d = 0.0, sum1=0.0, sum2=0.0;
    for(int i=0; i<length-4; i++){
        sum1 += vector1[i]*vector1[i];
        sum2 += vector2[i]*vector2[i];
    }
    for(int i=0; i<length-4; i++){
        d += (vector1[i]*vector2[i])/(sqrt(sum1)*sqrt(sum2));
        // std::cout<<log(vector1[i]/sum1)<<" "<<log(vector2[i]/sum2)<<std::endl;
    }
    return 1.0-d;
}

// Function to calculate the distance between two images
// Input argument isOne takes true for distance measure 1 and takes false for distance measure 2

void MainWindow::CalculateDistances(bool isOne)
{
    for(int n=0; n<num_images; n++) // for each image in the database
    {
        distances[n] = 0.0; // initialize to 0 the distance from query image to a database image

        for (int i = 0; i < queryfeature.size(); i++) // for each region in the query image
        {
            double current_distance = (double) RAND_MAX, new_distance;

            for (int j = 0; j < databasefeatures[n].size(); j++) // for each region in the current database image
            {
                if (isOne)
                    new_distance = distance1(queryfeature[i], databasefeatures[n][j], featurevectorlength);
                else
                    new_distance = distance2(queryfeature[i], databasefeatures[n][j], featurevectorlength);

                current_distance = std::min(current_distance, new_distance); // distance between the best matching regions
            }

            distances[n] = distances[n] + current_distance; // sum of distances between each matching pair of regions
        }

        distances[n] = distances[n] / (double) queryfeature.size(); // normalize by number of matching pairs

        // Display the distance values
        ui->progressBox->append(QString::fromStdString("Distance to image "+std::to_string(n+1)+" = "+std::to_string(distances[n])));
        QApplication::processEvents();
    }
}
