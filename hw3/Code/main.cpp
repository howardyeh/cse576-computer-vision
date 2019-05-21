#include <QtWidgets/QApplication>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    /***** Use the following 2 lines to scale up the application window *****/

    //qputenv("QT_AUTO_SCREEN_SCALE_FACTOR", "1");
    //qputenv("QT_SCALE_FACTOR", "0.75"); // use a suitable scaling factor

    const int RESTART_CODE = 1000;
    int return_code;
    do
    {
        QApplication a(argc, argv);
		
		/***** Use the following 3 lines to change the text font size (here 10) *****/
		/***** You may need to increase the application window size first and then decrease the font size to make things look better *****/
		
		//QFont f = a.font();
		//f.setPointSize(10);
		//a.setFont(f);    	
	
        MainWindow w;
        w.show();
        return_code = a.exec();
    }while (return_code == RESTART_CODE);

    return return_code;
}
