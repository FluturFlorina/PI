// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void negative_imagine()
{
	//implement function
	Mat img = imread("Images/cameraman.bmp",
		IMREAD_GRAYSCALE);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
		}
	}
	imshow("negative image", img);
	waitKey(0);
}

void nivele_gri(int additiveFactor)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				//uchar neg = 255 - val;
				int newVal = val + additiveFactor;
				if (newVal > 255)
					newVal = 255;
				//uchar neg = 255 - newVal;
				dst.at<uchar>(i, j) = newVal;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}


void factor_multiplicativ(int additiveFactor)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				//uchar neg = 255 - val;
				int newVal = val * additiveFactor;
				if (newVal > 255)
					newVal = 255;
				//uchar neg = 255 - newVal;
				dst.at<uchar>(i, j) = newVal;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("gri image", dst);
		imwrite("imaginegri.jpg", dst);
		waitKey();
	}
}

void imagine_color()
{
	Mat img(256, 256, CV_8UC3);
	int height = img.rows;
	int width = img.cols;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (i <128 && j <128)
				img.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			else if(i <128 && j >128)
				img.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
			else if (i > 128 && j <128)
				img.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
			else if (i > 128 && j >128)
				img.at<Vec3b>(i, j) = Vec3b(0, 255, 255);
		}
	}
	imshow("Imagine color", img);
	waitKey();
}


void inversa()
{
	float vals[9] = { 1,-1,1,2,0,3,1,1,-2 };
	Mat M(3, 3, CV_32FC1, vals);

	//pas1 determinant
	float det = M.at<float>(0, 0) * (M.at<float>(1, 1) * M.at<float>(2, 2) - M.at<float>(1, 2) * M.at<float>(2, 1)) -M.at<float>(0, 1) * (M.at<float>(1, 0) * M.at<float>(2, 2) - M.at<float>(1, 2) * M.at<float>(2, 0)) +M.at<float>(0, 2) * (M.at<float>(1, 0) * M.at<float>(2, 1) - M.at<float>(1, 1) * M.at<float>(2, 0));
	if (det == 0)
	{
		std::cout << "Nu exista inversa matricei";
		return;
	}

	//pas2 transpusa
	float val1[9] = { vals[8],vals[7],vals[6],vals[5],vals[4],vals[3],vals[2],vals[1],vals[0] };
	Mat adj(3, 3, CV_32FC1, val1);

	//pas3 adjuncta
	
	float a, b, c, d, e, f, g, h, i;
	a = adj.at<float>(1, 1) * adj.at<float>(2, 2) - adj.at<float>(2, 1) * adj.at<float>(1, 2);
	b = -(adj.at<float>(1, 0) * adj.at<float>(2, 2) - adj.at<float>(2, 0) * adj.at<float>(1, 2));
	c = adj.at<float>(1, 0) * adj.at<float>(2, 1) - adj.at<float>(2, 0) * adj.at<float>(1, 1);
	d = -(adj.at<float>(0, 1) * adj.at<float>(2, 2) - adj.at<float>(2, 1) * adj.at<float>(0, 2));
	e = adj.at<float>(0, 0) * adj.at<float>(2, 2) - adj.at<float>(2, 0) * adj.at<float>(0, 2);
	f = -(adj.at<float>(0, 0) * adj.at<float>(2, 1) - adj.at<float>(2, 0) * adj.at<float>(0, 1));
	g = adj.at<float>(0, 1) * adj.at<float>(1, 2) - adj.at<float>(1, 1) * adj.at<float>(0, 2);
	h = -(adj.at<float>(0, 0) * adj.at<float>(1, 2) - adj.at<float>(1, 0) * adj.at<float>(0, 2));
	i = adj.at<float>(0, 0) * adj.at<float>(1, 1) - adj.at<float>(1, 0) * adj.at<float>(0, 1);


	float val2[9] = { a,b,c,d,e,f,g,h,i };
	Mat adj2(3, 3, CV_32FC1, val2);


	M = (1 / det) * adj2;

	std::cout << M<< std::endl;
	//waitKey(10000000000000000000);
}

void matrici()
{
	
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		Mat MR(height, width, CV_8UC3);
		Mat MG(height, width, CV_8UC3);
		Mat MB(height, width, CV_8UC3);
		
		
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);
				MB.at<Vec3b>(i, j)= Vec3b(pixel[2], 0, 0);
				MG.at<Vec3b>(i, j) = Vec3b(0, pixel[1], 0);
				MR.at<Vec3b>(i, j) = Vec3b(0, 0, pixel[0]);
			}
		}

		//imshow("input image", src);
		//imshow("gri image", dst);
		imshow("R", MR);
		imshow("G", MG);
		imshow("B", MB);
		imshow("", NULL);
		//imwrite("imaginegri.jpg", dst);
		waitKey();
	}

}

void grayscale()
{

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		Mat M(height, width, CV_8UC1);
		


		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);
				uchar med = (pixel[2] + pixel[1] + pixel[0]) / 3;
				M.at<uchar>(i, j) = med;
				
			}
		}

		imshow("input image", src);
		//imshow("gri image", dst);
		imshow("GRAY", M);
		//imshow("G", MG);
		//imshow("B", MB);
		//imwrite("imaginegri.jpg", dst);
		waitKey();
	}

}

void albnegru(int prag)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat M(height, width, CV_8UC1);



		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar pixel = src.at<uchar>(i, j);
				int p = prag;
				if(pixel < prag)
					M.at<uchar>(i, j) = 0;
				else
					M.at<uchar>(i, j) = 255;

			}
		}

		imshow("input image", src);
		//imshow("gri image", dst);
		imshow("AlbNegru", M);
		//imshow("G", MG);
		//imshow("B", MB);
		//imwrite("imaginegri.jpg", dst);
		waitKey();
	}
}

void rgb24()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{

		Mat src = imread(fname,CV_LOAD_IMAGE_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat H(height, width, CV_8UC1);
		Mat S(height, width, CV_8UC1);
		Mat V(height, width, CV_8UC1);



		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b pixel = src.at<Vec3b>(i, j);
				float r = pixel[2]/255.0;
				float g = pixel[1]/255.0;
				float b = pixel[0]/255.0;

				float M1 = max(r, g);
				float M = max(M1, b);
				float m1 = min(r, g);
				float m = min(m1, b);

				float C = M - m;
				float V1 = M;
				float S1;

				if (V1 != 0)
					S1 = C / V1;
				else
					S1 = 0;

				float H1;
				if (C != 0)
				{
					if (M1 == r)
						H1 = 60 * (g - b) / C;
					if (M1 == g)
						H1 = 120 + 60 * (b - r) / C;
					if (M1 == b)
						H1 = 240 + 60 * (r - g) / C;
				}
				else
					H1 = 0;
				if (H1 < 0)
					H1 = H1 + 360;

				H.at<uchar>(i,j)= (H1 * 255 / 360);
				S.at<uchar>(i, j) =(S1 * 255);
				V.at<uchar>(i, j) = (V1 * 255);

			}
		}

		imshow("input image", src);
		//imshow("gri image", dst);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);
		//imshow("G", MG);
		//imshow("B", MB);
		//imwrite("imaginegri.jpg", dst);
		waitKey();
	}
}

Mat citire_img()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		imshow("input image", src);
		return src;
		waitKey();
	}
}

bool isInside(Mat img,uchar i,uchar j)
{
	int height = img.rows;
	int width = img.cols;
	int ok = 0;

	for (int x = 0; x < height; x++)
	{
		for (int y = 0; y < width; y++)
		{
			if (x == i)
				ok++;
			if (y == j)
				ok++;
		}
	}
	if (ok == 2)
		return true;
	else
		return false;
}

int main()
{
	int op;
	int fac = 0;
	int i, j;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - L1 Negative Image \n");
		//printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 -Gri cu factor aditional\n");
		printf(" 11 -Gri cu factor multiplicativ\n");
		printf(" 12 -Imagine color\n");
		printf(" 13 -Inversa unei matrici\n");
		//lab2
		printf(" 14 -3 Matrici diferite\n");
		printf(" 15 -Grayscale\n");
		printf(" 16 -Alb-Negru\n");
		printf(" 17 -RGB24\n");
		printf(" 18 -isInside\n");

		//lab3

		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			/*case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				negative_imagine();
				break;
			case 0:
				testVideoSequence();
				break;
				
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				std::cout << "Introduceti factorul aditional\n";
				std::cin >> fac;
				nivele_gri( fac);
				break;
			case 11:
				std::cout << "Introduceti factorul multiplicativ\n";
				std::cin >> fac;
				factor_multiplicativ(fac);
				break;
			case 12:
				imagine_color();
				break;
			case 13:
				inversa();
				break;*/
			case 14:
				matrici();
				break;
			case 15:
				grayscale();
				break;
			case 16:
				std::cin >> fac;
				albnegru(fac);
				break;
			case 17:
				rgb24();
				break;
			case 18:
				std::cin >> i>> j;
				Mat img = citire_img();
				std::cout<<isInside(img,i,j);
				imshow("", NULL);
				waitKey();
				break;

		}
	}
	while (op!=0);
	return 0;
}
