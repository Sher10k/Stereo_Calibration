/*
 *  Calibration algorithm 
 * 
 * 1) Load image FLZcmCameraBaslerJpegFrame*.png
 *           and FRZcmCameraBaslerJpegFrame*.png
 * 
 * 2) Find couples of images
 * 
 * 3) Chessboard & Charuco calibration
 * 
 *  
*/




#include <QCoreApplication>
#include <iostream>
#include <fstream>
#include <set>                  // заголовочный файл множеств и мультимножеств
#include <deque>                // Деки
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <getopt.h>
#include <string>

#include <Eigen/Eigen>
#include <Open3D/Open3D.h>
#include <Open3D/Geometry/Geometry.h>
#include <Open3D/Geometry/Geometry3D.h>
#include <Open3D/Geometry/PointCloud.h>
#include <Open3D/Geometry/Octree.h>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace open3d;

#define FRAME_WIDTH 2064    //2448
#define FRAME_HEIGHT 1544   //2048
#define BOARD_X 11
#define BOARD_Y 8

// --- GLOBAL VARIABLES -----------------------------------------------------//
static Size imageSize = Size(FRAME_WIDTH, FRAME_HEIGHT);

    // ChArUco board variables
static Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary( aruco::DICT_6X6_250 );  // DICT_6X6_250 = 10
    // create charuco board object
static Ptr<aruco::CharucoBoard> charucoboard = aruco::CharucoBoard::create(BOARD_X, BOARD_Y, 0.03f, 0.02f, dictionary);

static TermCriteria termcritSub( TermCriteria::EPS | TermCriteria::MAX_ITER, 100, 0.0001 );

static int Calibration_flags = CALIB_FIX_K1 | 
                              CALIB_FIX_K2 | 
                              CALIB_FIX_K3;      // Calibration flags    | CALIB_FIX_K6

static int Stereo_flag =  CALIB_FIX_INTRINSIC;
//                          CALIB_FIX_ASPECT_RATIO |
//                          CALIB_ZERO_TANGENT_DIST |
//                          CALIB_USE_INTRINSIC_GUESS |
//                          CALIB_SAME_FOCAL_LENGTH |
//                          CALIB_RATIONAL_MODEL |
//                          CALIB_FIX_K3 | 
//                          CALIB_FIX_K4 | 
//                          CALIB_FIX_K5;

    // ChessBoard size
static int _numCornersHor = 8;  // 4; 
static int _numCornersVer = 5;  // 3;
static Size board_sz = Size( _numCornersHor, _numCornersVer ); 


struct Args
{
    std::string infolder    = "";
    std::string board       = "";
    std::string patternSize = "";
    std::string mod         = "hand";

    bool parse( int argc, char *argv[] )
    {
        // set some defaults
        const char *optstring = "i:b:p:m:h";
        struct option long_opts[] = {
                { "input",       required_argument, nullptr, 'i' },
                { "board",       required_argument, nullptr, 'b' },
                { "patternSize", required_argument, nullptr, 'p' },
                { "mod",         optional_argument, nullptr, 'm' },
                { "help",        no_argument,       nullptr, 'h' },
                { nullptr,       0,                 nullptr,  0  }
        };

        int c;
        while ( ( c = getopt_long ( argc, argv, optstring, long_opts, nullptr ) ) >= 0 ) {
            switch (c) {
                case 'i': infolder      = std::string(optarg);  break;
                case 'b': board         = std::string(optarg);  break;
                case 'p': patternSize   = std::string(optarg);  break;
                case 'm': mod           = std::string(optarg);  break;
                case 'h': default: usage(); return false;
            }
        }

        if ( infolder == "" ) {
            std::cerr << " ! ! ! Please specify a folder with stereo frames" << std::endl;
            return false;
        
        }
        if ( board  == "" ) {
            std::cerr << " ! ! ! Please specify the type of calibration board" << std::endl;
            return false;
        }
        if ( patternSize  == "" ) {
            std::cerr << " ! ! ! Please specify the number of internal corners per "
                         "row and column or the number of squares of a checkerboard" << std::endl;
            return false;
        }
        
        cout << "Infolder: \t" << infolder << endl;
        cout << "Board: \t\t" << board << endl;
        cout << "PatternSize: \t" << patternSize << endl;
        cout << "Mod: \t\t" << mod << endl;
        
        //dictionary = aruco::getPredefinedDictionary( aruco::DICT_6X6_250 );
        
        return true;
    }

    void usage()
    {
        std::cout << "Usage: ./zcm2video [options]" << std::endl
                  << "" << std::endl
                  << "    Convert zcm log file to stereo img files" << std::endl
                  << "    and get list zcm event parametrs" << std::endl
                  << "" << std::endl
                  << "Example:" << std::endl
                  << "    ./zcm2video -i ../zcm.log -o ../stereoIMG -p 5" << std::endl
                  << "" << std::endl
                  << "And view list parametrs: " << std::endl
                  << "    ./zcm2video -i ../zcm.log -l " << std::endl
                  << "" << std::endl
                << "Options:" << std::endl
                << "" << std::endl
                << "  -h, --help                  Shows this help text and exits" << std::endl
                << "  -i, --input = logfile       Input log to convert" << std::endl
                << "  -o, --output = outputfolder Output stereo imgs folder" << std::endl
                << "  -p, --param = parameters    Input parameters number of frames" << std::endl
                << "  -l, --list = list           Output list parametrs" << std::endl
                << "  -d, --debug                 Run a dry run to ensure proper converter setup" << std::endl
                << std::endl << std::endl;
    }
};



// --- FUNCTION ---------------------------------------------------------------------------------//

    // Prototype function
int get_dictionary( string str_board );                             // Get number charuco board
vector< float > parser_pat( string patternSize );                   // Board pattern size parser

void read_file_image( vector < string > (&imgPath)[2], Args &args );                 // Read image path

int detectedChessBoard( vector < string > (&imgPath)[2],                 // Detecting checkerboard corners
                        vector< vector< Point2f > > (&allCorners)[2],    // and erasing intersecting boards (threshold 80%)
                        vector < float > &patternSize );


void read_CharucoBoard( vector < string > * imgPath,
                        vector < vector< vector< Point2f > > > * allCorners, 
                        vector < vector< int > > * allIds, 
                        vector < Mat > * allCharucoCorners, 
                        vector < Mat > * allCharucoIds, 
                        set < unsigned int > * nGoodboard )
{
    for (unsigned int i = 0; i < imgPath->size(); i++)                        // Цикл для определенного числа калибровочных кадров img->size()
    {
        //Mat imgi = img->at(i);
        Mat imgi = imread( imgPath->at(i) );
        imgi(Rect(0, 0, 2448, 2048)).copyTo(imgi);
        //rotate(imgi, imgi, ROTATE_180);
        Mat imgGray;
        cvtColor( imgi, imgGray, COLOR_BGR2GRAY );
        
        vector< vector< Point2f > > corners;
        vector< int > ids;
        
            // Detect markers
        aruco::detectMarkers( imgGray, 
                              dictionary, 
                              corners, 
                              ids, 
                              aruco::DetectorParameters::create());
        
        if ( ids.size() == (BOARD_X * BOARD_Y) / 2 )                          // Проверка удачно найденых углов == 44
        {
                // SUB PIXEL DETECTION
            for (size_t j = 0; j < corners.size(); j++)
            {
//                cornerSubPix( imgGray, 
//                              corners[j], 
//                              Size(20, 20), 
//                              Size(-1, -1), 
//                              termcritSub);
                cornerSubPix( imgGray, 
                              corners[j], 
                              Size(5, 5), 
                              Size(-1, -1), 
                              termcritSub);
            }
                // Interpolate charuco corners
            Mat charucoCorners, charucoIds;
            aruco::interpolateCornersCharuco( corners, 
                                              ids, 
                                              imgGray, 
                                              charucoboard, 
                                              charucoCorners,
                                              charucoIds);
            allCorners->push_back(corners);
            allIds->push_back(ids);
            allCharucoCorners->push_back(charucoCorners);
            allCharucoIds->push_back(charucoIds);
            
            //nGoodboard->push_back(i);
            nGoodboard->insert(i);
        }
    } 
}

void read_Chessboards( vector < string > * imgPath,
                       vector< vector< Point2f > > * allCorners,
                       set < unsigned int > * nGoodboard )
{
    for (unsigned int i = 0; i < imgPath->size(); i++)                        // Цикл для определенного числа калибровочных кадров img->size()
    {
        Mat imgi = imread( imgPath->at(i) );
        Mat imgGray;
        cvtColor( imgi, imgGray, COLOR_BGR2GRAY );
        
        vector< Point2f > corners;
        
            // Find cernels on chessboard
        bool found = findChessboardCorners( imgGray,
                                            board_sz,
                                            corners,
                                            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE ); //CALIB_CB_NORMALIZE_IMAGE, CV_CALIB_CB_FILTER_QUADS
        
        if (found)    // Проверка удачно найденых углов
        {
//            cornerSubPix( imgGray,
//                          corners,
//                          Size(11,11),
//                          Size(-1,-1),
//                          TermCriteria( TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1 ) );   // Уточнение углов
            
            allCorners->push_back(corners);
            
            //nGoodboard->push_back(i);
            nGoodboard->insert(i);
        }
    } 
}

void output_stereo_img( Mat * img,
                        string * imgPath,
                        vector < vector < Point2f > > * Corners, 
                        Mat * CharucoCorners,
                        Mat * CharucoIds,
                        Mat rmapX,
                        Mat rmapY )
{
    Mat imgi = imread( *imgPath );
    imgi(Rect(0, 0, 2448, 2048)).copyTo( *img );
    //rotate(*img, *img, ROTATE_180);
    Mat tempF;
    
    aruco::drawDetectedMarkers( *img, 
                                *Corners );
    aruco::drawDetectedCornersCharuco( *img, 
                                       *CharucoCorners, 
                                       *CharucoIds);
//    undistort( *img, tempF, 
//               cameraMatrix[0], 
//               distCoeffs[0] );
    img->copyTo(tempF);
    remap( tempF, *img, 
           rmapX, rmapY, 
           INTER_LINEAR );
}

void output_stereo_img_chessboard( Mat * img,
                                   string * imgPath,
                                   vector < Point2f > * Corners,
                                   Matx33d cameraMatrix, 
                                   Matx < double, 1, 5 > distCoeffs,
                                   Mat rmapX,
                                   Mat rmapY )
{
    Mat imgi = imread( *imgPath );
    imgi.copyTo( *img );
    Mat tempF;
    
    drawChessboardCorners( *img,
                           board_sz,
                           *Corners, 
                           true ); 
    undistort( *img, tempF, 
               cameraMatrix, 
               distCoeffs );
    remap( tempF, *img, 
           rmapX, rmapY, 
           INTER_LINEAR );
}

// --- MAIN -------------------------------------------------------------------------------------//
int main( int argc, char *argv[] )  //int argc, char *argv[]
{
    
//    Mat img = imread("ChArUcoBoard.png", IMREAD_COLOR);
//    resize(img, img, Size(640, 480));
//    imshow("Output", img);
    
// --- INPUT PARAMETERS -------------------------------------------------------------------------//
    Args args;
    if ( !args.parse( argc, argv ) ) return 1;
    
// --- MAIN VARIABLES ---------------------------------------------------------------------------//
    vector < string > imgPath[2];                                           // FILES IMAGE
    FileStorage fs;
    fs.open( "Stereo_calib_RESULT.txt", FileStorage::WRITE );               // Text file for write
    
// --- STEP 1 --- Load left and right frames ----------------------------------------------------//
    read_file_image( imgPath, args );
    
    
// --- STEP 2 --- Calibration left and right camera ---------------------------------------------//
        // ChessBoard calibration
    if ( (args.board == "chess") || (args.board == "Chess") || (args.board == "CHESS") || 
         (args.board == "Chessboard") || (args.board == "ChessBoard") || (args.board == "CHESSBOARD") )
    {
        vector < float > patternSize;                   // H x W x Size_square x Size_marker
        patternSize = parser_pat( args.patternSize );
        
        vector< vector< Point2f > > allCorners[2];      // All corners of chess board
        //set < unsigned > nGoodboard[2];                 // Well found corners of chess board
        
        detectedChessBoard( imgPath, allCorners, patternSize );
        
    }
        // Charuco calibration
    else
    {
        
    }
    vector< vector< Point2f > > allCornersL;
    vector< vector< Point2f > > allCornersR;
    set < unsigned int > nGoodboard[2];
    read_Chessboards( & imgPath[0],
                      & allCornersL,
                      & nGoodboard[0]);
    read_Chessboards( & imgPath[1],
                      & allCornersR,
                      & nGoodboard[1]);
    cout << "Элементы множества nGoodboard[0]: ";
    copy( nGoodboard[0].begin(), nGoodboard[0].end(), ostream_iterator<int>(cout, " "));
    cout << endl;
    cout << "Элементы множества nGoodboard[1]: ";
    copy( nGoodboard[1].begin(), nGoodboard[1].end(), ostream_iterator<int>(cout, " "));
    cout << endl;
    vector < unsigned int > nGoodboardv;
    set_intersection( nGoodboard[0].begin(), nGoodboard[0].end(), 
                      nGoodboard[1].begin(), nGoodboard[1].end(), 
                      back_inserter(nGoodboardv));
    cout << "Соответствующие пары nGoodboardv: ";
    for (size_t i = 0; i < nGoodboardv.size(); i++) cout << nGoodboardv[i] << " ";
    cout << endl << "nGoodboardv = " << nGoodboardv.size() << endl;
    
        // Sorting the relevant elements
    vector < unsigned int > nGL(nGoodboard[0].size());
    copy( nGoodboard[0].begin(), nGoodboard[0].end(), nGL.begin() );
    vector < unsigned int > nGR(nGoodboard[1].size());
    copy( nGoodboard[1].begin(), nGoodboard[1].end(), nGR.begin() );
    for ( unsigned int i = 0, j = 0, k = 0; k < nGoodboardv.size() ; )
    {
        if ( nGL.at(i) == nGR.at(j) )
        {
            k++;
            i++;
            j++;
        }
        else if ( nGL.at(i) != nGoodboardv.at(k))
        {
            nGL.erase(nGL.begin() + i);
            allCornersL.erase(allCornersL.begin() + i);
        }
        else if ( nGR.at(j) != nGoodboardv.at(k))
        {
            nGR.erase(nGR.begin() + j);
            allCornersR.erase(allCornersR.begin() + j);
        }
        else
        {
            nGL.erase(nGL.begin() + i);
            allCornersL.erase(allCornersL.begin() + i);
            
            nGR.erase(nGR.begin() + j);
            allCornersR.erase(allCornersR.begin() + j);
        }
    }
    nGL.resize(nGoodboardv.size());
    nGR.resize(nGoodboardv.size());
    allCornersL.resize(nGoodboardv.size());
    allCornersR.resize(nGoodboardv.size());
    
// --- Read camera internal settings
    cout << endl << " --- --- READ camera options" << endl;
    Matx < double, 3, 3 > cameraMatrix[2];
    Matx < double, 1, 5 > distCoeffs[2];
    string parametersDir = "/home/roman/Reconst_Stereo/calibration_parameters/initrics/22500062/2019.07.02/";
    
// --- Left camera options
    cout << " --- LEFT camera" << endl;
        // Camera matrix
    fstream file_params( parametersDir + "mtx.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "mtx.csv" << endl;
        exit(0);
    }
    for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++)
            file_params >> cameraMatrix[0](i, j);
    cout << "cameraMatrixL = " << endl << cameraMatrix[0] << endl;
    file_params.close();
        // Distortion matrix
    file_params.open( parametersDir + "dist.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "dist.csv" << endl;
        exit(0);
    }
    for ( int j = 0; j < 5; j++ )
        file_params >> distCoeffs[0](0, j);
    cout << "distCoeffsL = " << endl << distCoeffs[0] << endl;
    file_params.close();
    
// --- Right camera options
    cout << endl << " --- RIGHT camera" << endl;
        // Camera matrix
    file_params.open( parametersDir + "to_right/22500061/mtxR.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "to_right/22500061/mtxR.csv" << endl;
        exit(0);
    }
    for ( int i = 0; i < 3; i++ )
        for ( int j = 0; j < 3; j++)
            file_params >> cameraMatrix[1](i, j);
    cout << "cameraMatrixR = " << endl << cameraMatrix[1] << endl;
    file_params.close();
        // Distortion matrix
    file_params.open( parametersDir + "to_right/22500061/distR.csv" );
    if ( !file_params.is_open() )
    {
        cout << " --- file_params not open: " << parametersDir + "to_right/22500061/distR.csv" << endl;
        exit(0);
    }
    for ( int j = 0; j < 5; j++ )
        file_params >> distCoeffs[1](0, j);
    cout << "distCoeffsR = " << endl << distCoeffs[1] << endl;
    file_params.close();
    cout << " --- --- END READ camera options" << endl;
    
    // Zero distors---------------------------------------------------!!!!!!!!!!!!!!!!!
    for ( int i = 0; i < 2; i++ )
        for ( int j = 0; j < 5; j++ )
            distCoeffs[i](0, j) = 0;
    
// --- END Read camera internal settings
    
    
        // Collect data from each frame
/*    vector < vector< vector< Point2f > > > allCorners[2];
    vector < vector< int > > allIds[2];
    vector < Mat > allCharucoCorners[2];
    vector < Mat > allCharucoIds[2];
    
    set < unsigned int > nGoodboard[2];
    read_CharucoBoard( & imgPath[0],
                       & allCorners[0], 
                       & allIds[0], 
                       & allCharucoCorners[0], 
                       & allCharucoIds[0], 
                       & nGoodboard[0] );
    read_CharucoBoard( & imgPath[1],
                       & allCorners[1], 
                       & allIds[1], 
                       & allCharucoCorners[1], 
                       & allCharucoIds[1], 
                       & nGoodboard[1] );
    cout << "Элементы множества nGoodboard[0]: ";
    copy( nGoodboard[0].begin(), nGoodboard[0].end(), ostream_iterator<int>(cout, " "));
    cout << endl;
    cout << "Элементы множества nGoodboard[1]: ";
    copy( nGoodboard[1].begin(), nGoodboard[1].end(), ostream_iterator<int>(cout, " "));
    cout << endl;
    vector < unsigned int > nGoodboardv;
    set_intersection( nGoodboard[0].begin(), nGoodboard[0].end(), 
                      nGoodboard[1].begin(), nGoodboard[1].end(), 
                      back_inserter(nGoodboardv));
    cout << "Соответствующие пары nGoodboardv: ";
    for (size_t i = 0; i < nGoodboardv.size(); i++) cout << nGoodboardv[i] << " ";
    cout << endl << "nGoodboardv = " << nGoodboardv.size() << endl;
    
        // Sorting the relevant elements
    vector < unsigned int > nGL(nGoodboard[0].size());
    copy( nGoodboard[0].begin(), nGoodboard[0].end(), nGL.begin() );
    vector < unsigned int > nGR(nGoodboard[1].size());
    copy( nGoodboard[1].begin(), nGoodboard[1].end(), nGR.begin() );
    for ( unsigned int i = 0, j = 0, k = 0; k < nGoodboardv.size() ; )
    {
        if ( nGL.at(i) == nGR.at(j) )
        {
            k++;
            i++;
            j++;
        }
        else if ( nGL.at(i) != nGoodboardv.at(k))
        {
            nGL.erase(nGL.begin() + i);
            allCorners[0].erase(allCorners[0].begin() + i);
            allIds[0].erase(allIds[0].begin() + i);
            allCharucoCorners[0].erase(allCharucoCorners[0].begin() + i);
            allCharucoIds[0].erase(allCharucoIds[0].begin() + i);
        }
        else if ( nGR.at(j) != nGoodboardv.at(k))
        {
            nGR.erase(nGR.begin() + j);
            allCorners[1].erase(allCorners[1].begin() + j);
            allIds[1].erase(allIds[1].begin() + j);
            allCharucoCorners[1].erase(allCharucoCorners[1].begin() + j);
            allCharucoIds[1].erase(allCharucoIds[1].begin() + j);
        }
        else
        {
            nGL.erase(nGL.begin() + i);
            allCorners[0].erase(allCorners[0].begin() + i);
            allIds[0].erase(allIds[0].begin() + i);
            allCharucoCorners[0].erase(allCharucoCorners[0].begin() + i);
            allCharucoIds[0].erase(allCharucoIds[0].begin() + i);
            
            nGR.erase(nGR.begin() + j);
            allCorners[1].erase(allCorners[1].begin() + j);
            allIds[1].erase(allIds[1].begin() + j);
            allCharucoCorners[1].erase(allCharucoCorners[1].begin() + j);
            allCharucoIds[1].erase(allCharucoIds[1].begin() + j);
        }
    }
    nGL.resize(nGoodboardv.size());
    nGR.resize(nGoodboardv.size());
    for (unsigned int n = 0; n <2; n++)
    {
        allCorners[n].resize(nGoodboardv.size());
        allIds[n].resize(nGoodboardv.size());
        allCharucoCorners[n].resize(nGoodboardv.size());
        allCharucoIds[n].resize(nGoodboardv.size());
    }
    
        // Calibration left and right camera
    Matx33d cameraMatrix[2];
    Matx< double, 1, 5 > distCoeffs[2];
    vector< Mat > rvecs[2], tvecs[2];
    
    calibrateCameraCharuco( allCharucoCorners[0],
                            allCharucoIds[0], 
                            charucoboard, 
                            imageSize,
                            cameraMatrix[0],
                            distCoeffs[0],
                            rvecs[0],
                            tvecs[0],
                            Calibration_flags);
    calibrateCameraCharuco( allCharucoCorners[1],
                            allCharucoIds[1], 
                            charucoboard, 
                            imageSize,
                            cameraMatrix[1],
                            distCoeffs[1],
                            rvecs[1],
                            tvecs[1],
                            Calibration_flags);
    
    fs << "intrinsicL" << cameraMatrix[0];
    fs << "distCoeffsL" << distCoeffs[0];
    fs << "rvecsL" << rvecs[0];
    fs << "tvecsL" << tvecs[0];
    fs << "intrinsicR" << cameraMatrix[1];
    fs << "distCoeffsR" << distCoeffs[1];
    fs << "rvecsR" << rvecs[1];
    fs << "tvecsR" << tvecs[1];*/
    
    fs << "intrinsicL" << cameraMatrix[0];
    fs << "distCoeffsL" << distCoeffs[0];
    fs << "intrinsicR" << cameraMatrix[1];
    fs << "distCoeffsR" << distCoeffs[1];
    
// --- STEP 3 --- Stereo calibration ----------------------------------------//
    vector < vector < Point3f > > objectPoints;
//    objectPoints.resize(nGoodboardv.size());
//    int squareSize = 1;
//    for(unsigned int i = 0; i < nGoodboardv.size(); i++ )
//    {
//        for( int j = 0; j < BOARD_X - 1; j++ )
//            for( int k = 0; k < BOARD_Y - 1; k++ )
//                objectPoints[i].push_back( Point3f( k * squareSize, j * squareSize, 0 ) );
//    }
    
    int numSquares = _numCornersHor * _numCornersVer;
    double square_size = 1.0;    
    vector< Point3f > obj;
    for(int j = 0; j < numSquares; j++)
        obj.push_back( Point3d( j / _numCornersHor, j % _numCornersHor, 0.0 ) );
//    for( int i = 0; i < _numCornersVer; i++)
//        for( int j = 0; j < _numCornersHor; j++)
//            obj.push_back( Point3d( j * square_size, i * square_size, 0.0 ) );
    for(unsigned int i = 0; i < nGoodboardv.size(); i++ )
    {
        objectPoints.push_back( obj );
    }
    
    Mat R, T, E, F;

//    double rms = stereoCalibrate( objectPoints, 
//                                  allCharucoCorners[0], allCharucoCorners[1], 
//                                  cameraMatrix[0], distCoeffs[0],
//                                  cameraMatrix[1], distCoeffs[1],
//                                  imageSize, 
//                                  R, T, E, F,
//                                  Stereo_flag );
    double rms = stereoCalibrate( objectPoints, 
                                  allCornersL, allCornersR, 
                                  cameraMatrix[0], distCoeffs[0],
                                  cameraMatrix[1], distCoeffs[1],
                                  imageSize, 
                                  R, T, E, F,
                                  Stereo_flag );
    cout << "done with RMS error=" << rms << endl;
    fs << "R" << R;
    fs << "T" << T;
    fs << "E" << E;
    fs << "F" << F;
    fs << "RMS" << rms;
    
// --- STEP 4 --- Stereo rectify --------------------------------------------//
    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];
    stereoRectify( cameraMatrix[0], 
                   distCoeffs[0], 
                   cameraMatrix[1], 
                   distCoeffs[1], 
                   imageSize, 
                   R, T, R1, R2, P1, P2, Q, 
                   CALIB_ZERO_DISPARITY, 
                   -1, 
                   imageSize, 
                   &validRoi[0], 
                   &validRoi[1]);
    fs << "R1" <<R1;
    fs << "P1" <<P1;
    fs << "R2" <<R2;
    fs << "P2" <<P2;
    fs << "Q" <<Q;
    
// --- STEP 5 --- Undistort Rectify Map -------------------------------------//
    Mat rmap[2][2];
    initUndistortRectifyMap( cameraMatrix[0], 
                             distCoeffs[0], 
                             R1, P1, 
                             imageSize, 
                             CV_32FC1, 
                             rmap[0][0], rmap[0][1] );
    initUndistortRectifyMap( cameraMatrix[1], 
                             distCoeffs[1], 
                             R2, P2, 
                             imageSize, 
                             CV_32FC1,
                             rmap[1][0], rmap[1][1] );
    
// --- STEP 6 --- Undistort, Remap ------------------------------------------//
    for (size_t n = 0; n < nGoodboardv.size(); n++)
    {
        Mat imgL, imgR;
            // Left
//        output_stereo_img( & imgL,
//                           & imgPath[0][nGoodboardv[n]],
//                           & allCorners[0][n], 
//                           & allCharucoCorners[0][n],
//                           & allCharucoIds[0][n], 
//                           & cameraMatrix[0], 
//                           & distCoeffs[0], 
//                           rmap[0][0],
//                           rmap[0][1] );
        output_stereo_img_chessboard( & imgL,
                                      & imgPath[0][nGoodboardv[n]],
                                      & allCornersL[n],
                                      cameraMatrix[0],
                                      distCoeffs[0],
                                      rmap[0][0],
                                      rmap[0][1] );
            // Right
//        output_stereo_img( & imgR, 
//                           & imgPath[1][nGoodboardv[n]],
//                           & allCorners[1][n], 
//                           & allCharucoCorners[1][n],
//                           & allCharucoIds[1][n], 
//                           & cameraMatrix[1], 
//                           & distCoeffs[1], 
//                           rmap[1][0],
//                           rmap[1][1] );
        output_stereo_img_chessboard( & imgR,
                                      & imgPath[1][nGoodboardv[n]],
                                      & allCornersR[n],
                                      cameraMatrix[1],
                                      distCoeffs[1],
                                      rmap[1][0],
                                      rmap[1][1] );
        
        Mat tempL, tempR;
        imgL.copyTo(tempL);
        imgR.copyTo(tempR);
            // --- DEPTH MAP
        Ptr < StereoSGBM > sbm = StereoSGBM::create( 0,                         // minDisparity
                                                     96,                       // numDisparities must be divisible by 16
                                                     17,                        // blockSize
                                                     0,                         // P1
                                                     2048,                         // P2              0
                                                     0,                         // disp12MaxDiff       1
                                                     0,                         // prefilterCap
                                                     0,                         // uniquenessRatio      10
                                                     0,                         // speckleWindowSize    100
                                                     0,                         // speckleRange          32
                                                     StereoSGBM::MODE_SGBM_3WAY );   // mode MODE_SGBM
//        resize( tempL, tempL, Size(1024,720), 0, 0, INTER_LINEAR );
//        resize( tempR, tempR, Size(1024,720), 0, 0, INTER_LINEAR );
        imwrite( "Remap_frame_L.png", tempL );
        imwrite( "Remap_frame_R.png", tempR );
        Mat imgGrey[2]; 
        cvtColor( tempL, imgGrey[0], COLOR_BGR2GRAY );
        cvtColor( tempR, imgGrey[1], COLOR_BGR2GRAY );
        
            // Calculate
        Mat imgDisp_bm;
        sbm->compute( imgGrey[0], imgGrey[1], imgDisp_bm );
        //sbm->compute( imgLine[0], imgLine[1], imgDisp_bm );
        //imwrite( "imgDisp_bm.png", imgDisp_bm );
        
        
            // Nomalization
        double minVal; double maxVal;
        minMaxLoc( imgDisp_bm, &minVal, &maxVal );
        Mat imgDispNorm_bm;
        imgDisp_bm.convertTo( imgDispNorm_bm, CV_8UC1, 255/(maxVal - minVal) );
        Mat imgDisp_color;
        applyColorMap( imgDispNorm_bm, imgDisp_color, COLORMAP_RAINBOW );   // COLORMAP_HOT
        imwrite( args.infolder + "ImgCalibStereo_ChessBoard_A0/Сali_pair_of_images_000" + to_string(n) + "_BM.png", imgDisp_color );
        cout << args.infolder + "ImgCalibStereo_ChessBoard_A0/Сali_pair_of_images_000" + to_string(n) + "_BM.png" << endl;
        
        
//            // Reprojects a disparity image to 3D space
//        Mat points3D;
//        reprojectImageTo3D( imgDispNorm_bm, points3D, Q, false );   // imgDispNorm_bm imgDisp_bm
        
//        auto pcl_ptr = make_shared< geometry::PointCloud >();
//        for ( size_t i = 0; i < points3D.total(); i++ )
//        {
//            Vector3d temPoint, tempColor;
//            temPoint.x() = static_cast< double >( points3D.at< Vec3f >( static_cast<int>(i) ).val[0] );
//            temPoint.y() = static_cast< double >( points3D.at< Vec3f >( static_cast<int>(i) ).val[1] );
//            temPoint.z() = static_cast< double >( points3D.at< Vec3f >( static_cast<int>(i) ).val[2] );
//            tempColor.x() = static_cast< double >( tempL.at< Vec3b >( static_cast<int>(i) ).val[2] );
//            tempColor.y() = static_cast< double >( tempL.at< Vec3b >( static_cast<int>(i) ).val[1] );
//            tempColor.z() = static_cast< double >( tempL.at< Vec3b >( static_cast<int>(i) ).val[0] );
//            pcl_ptr->points_.push_back( temPoint );
//            pcl_ptr->colors_.push_back( tempColor );
//        }
//        visualization::DrawGeometries( {pcl_ptr}, "Open3D", 1600, 900 );
        
        
        Mat frameLR = Mat::zeros(Size(2 * FRAME_WIDTH, FRAME_HEIGHT), CV_8UC3);
        Rect r1(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
        Rect r2(FRAME_WIDTH, 0, FRAME_WIDTH, FRAME_HEIGHT);
        putText( imgL, "L", Point(5, 140), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 10);
        putText( imgR, "R", Point(5, 140), FONT_HERSHEY_SIMPLEX, 5, Scalar(255, 0, 0), 10);
        imgL.copyTo(frameLR( r1 ));
        imgR.copyTo(frameLR( r2 ));
        for( int i = 0; i < frameLR.rows; i += 100 )
            for( int j = 0; j < frameLR.cols; j++ )
                frameLR.at< Vec3b >(i, j)[2] = 255;
        //imshow("calibration", frameLR);
//        imwrite( pos_dir + "/ImgCalibStereo/Сali_pair_of_images_000" + to_string(n) + ".png", frameLR);
//        cout << pos_dir + "/ImgCalibStereo/Сali_pair_of_images_000" + to_string(n) + ".png" << endl;
        imwrite( args.infolder + "ImgCalibStereo_ChessBoard_A0/Сali_pair_of_images_000" + to_string(n) + ".png", frameLR );
        cout << args.infolder + "ImgCalibStereo_ChessBoard_A0/Сali_pair_of_images_000" + to_string(n) + ".png" << endl;
    }
        
    fs.release();
    //cout << " --- Calibration data written into file: Stereo_calib_ChArUco.txt" << endl << endl;
    cout << " --- Calibration data written into file: Stereo_calib_ChessBoard_A0.txt" << endl << endl;
    
    //cin.get();
    waitKey(0);
    return 0;   // a.exec();
}

vector< float > parser_pat( string patternSize )
{
    vector < float > pS;
    std::string::size_type sz;
    string num;
    for ( auto i : patternSize )
    {
        if ( ((int(i) > 47) && (int(i) < 58)) || (int(i) == 46) )
        {
            num += i;
        }
        else
        {
            if ( !num.empty() ) 
            {
                pS.push_back( stof(num, &sz) );
                num.clear();
            }
        }
    }
    if ( !num.empty() ) 
        pS.push_back( stof(num, &sz) );
    return pS;
}

void read_file_image( vector < string > (&imgPath)[2], Args &args )
{
        // MENU
//    cout << "Input name_dir fo calib image: ";
//    cin >> pos_dir;
//    if (pos_dir == "0") pos_dir = "/home/roman/stereoIMG_ChessBoard_A0/"; //"/home/roman/imagesStereo/";
//    size_t kn = 2;
//    cout << "Which file to use: ";
//    cin >> kn;
//    if (kn <= 0) kn = 5;
    cout << "Start stereo calibration" << endl;
    
    string img_name_L = "FLZcmCameraBaslerJpegFrame*.png";
    string img_name_R = "FRZcmCameraBaslerJpegFrame*.png";
    string files_name_L = args.infolder + img_name_L;
    string files_name_R = args.infolder + img_name_R;
    vector < String > files_L, files_R;
    glob( files_name_L, files_L );
    glob( files_name_R, files_R );
    
    set < string > files_set_L, files_set_R;
    vector < string > files_set_LR;
    for ( size_t i = 0; i < files_L.size(); i++ ) 
    {
        string temp = files_L[i];
        files_L[i].erase(0, args.infolder.length() + 2);
        files_set_L.insert(files_L[i]);
        files_L[i] = temp;
    }
    for ( size_t i = 0; i < files_R.size(); i++ ) 
    {
        string temp = files_R[i];
        files_R[i].erase(0, args.infolder.length() + 2);
        files_set_R.insert(files_R[i]);
        files_R[i] = temp;
    }
        // Find pairs of stereo pair images
    set_intersection( files_set_L.begin(), files_set_L.end(), 
                      files_set_R.begin(), files_set_R.end(), 
                      back_inserter(files_set_LR));          // inserter(files_set_LR, files_set_LR.begin())
    size_t kn = 1;  // num frame
    if ( args.mod == "manual" || args.mod == "MANUAL" || 
         args.mod == "hand" || args.mod == "HAND" )
        kn = 1;
    else
        kn = size_t( stoi( args.mod ) );
    
    for ( size_t i = 0; i < files_set_LR.size(); i += kn )
    {
//        Rect myROI(0, 0, FRAME_HEIGHT, FRAME_WIDTH);            // 2048 x 2448
//        Mat imgL = imread( pos_dir + "FL" + files_set_LR[i] ); // load the image
//        Mat imgR = imread( pos_dir + "FR" + files_set_LR[i] );
        imgPath[0].push_back( args.infolder + "FL" + files_set_LR[i] );
        imgPath[1].push_back( args.infolder + "FR" + files_set_LR[i] );
//        imgL(Rect(0, 0, 2448, 2048)).copyTo(imgL);
//        imgR(Rect(0, 0, 2448, 2048)).copyTo(imgR);
//        rotate(imgL, imgL, ROTATE_180);
//        rotate(imgR, imgR, ROTATE_180);
    }
}

int detectedChessBoard( vector < string > (&imgPath)[2], 
                        vector< vector< Point2f > > (&allCorners)[2],
                        vector < float > &patternSize )
{
    Size board_sz = Size( int(patternSize[0]), int(patternSize[1]) );
    if ( imgPath[0].size() != imgPath[1].size() ) 
    {
        cout << "ImgPath sizes are not equal" << endl;
        exit(0);
    }
    int click = 0;
    for ( unsigned i = 0; i < imgPath[0].size(); i++ )                        // Цикл для определенного числа калибровочных кадров img->size()
    {
        Mat imgiL = imread( imgPath[0].at(i) );
        Mat imgiR = imread( imgPath[1].at(i) );
        Mat imgGrayL, imgGrayR;
        cvtColor( imgiL, imgGrayL, COLOR_BGR2GRAY );
        cvtColor( imgiR, imgGrayR, COLOR_BGR2GRAY );
        
            // Find cernels on chessboard
        vector< Point2f > cornersL, cornersR;
        bool foundL = findChessboardCorners( imgGrayL,
                                             board_sz,
                                             cornersL,
                                             CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE ); //CALIB_CB_NORMALIZE_IMAGE, CV_CALIB_CB_FILTER_QUADS
        bool foundR = findChessboardCorners( imgGrayR,
                                             board_sz,
                                             cornersR,
                                             CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE );
        
        if ( foundL && foundR )    // Проверка удачно найденых углов
        {
//            cornerSubPix( imgGray,
//                          corners,
//                          Size(11,11),
//                          Size(-1,-1),
//                          TermCriteria( TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1 ) );   // Уточнение углов
            Mat final = Mat::zeros( imgiL.size(), CV_8UC3 );
            Mat mask = Mat::zeros( imgiL.size(), CV_8UC1 );
            vector< vector< Point > > vpts;
            vector< Point > pts;
            pts.push_back( cornersL.at( 0 ) );
            pts.push_back( cornersL.at( unsigned(patternSize[0]) - 1 ) );
            pts.push_back( cornersL.at( unsigned(patternSize[0]) * unsigned(patternSize[1]) - 1) );
            pts.push_back( cornersL.at( unsigned(patternSize[0]) * ( unsigned(patternSize[1]) - 1 )) );
            vpts.push_back( pts );
            fillPoly( mask, vpts, Scalar(255, 255, 255), 8, 0 );
            bitwise_and( imgiL, imgiL, final, mask );
            //waitKey(0);
            
            
            drawChessboardCorners( imgiL,
                                   board_sz,
                                   cornersL, 
                                   true ); 
            drawChessboardCorners( imgiR,
                                   board_sz,
                                   cornersR, 
                                   true );
            resize( imgiL, imgiL, Size(640,480), 0, 0, INTER_LINEAR );
            resize( imgiR, imgiR, Size(640,480), 0, 0, INTER_LINEAR );
            resize( mask, mask, Size(640,480), 0, 0, INTER_LINEAR );
            resize( final, final, Size(640,480), 0, 0, INTER_LINEAR );
            imshow("Mask", mask);
            imshow("Result", final);
            //imshow("Source", imgiL);
            imshow("imgL", imgiL);
            //imshow("imgR", imgiR);
            cout << countNonZero(mask) << endl;
            allCorners[0].push_back(cornersR);
            
            click = waitKey(0);
            
            //nGoodboard->push_back(i);
            //nGoodboard[0].insert(i);
        }
        if( click == 27 ) break;                                // Interrupt the cycle, press "ESC"
    }
    return 0;
}

int get_dictionary( string str_board )
{
    if ( str_board == "DICT_4X4_50" ) return 0;
    if ( str_board == "DICT_4X4_100" ) return 1;
    if ( str_board == "DICT_4X4_250" ) return 2;
    if ( str_board == "DICT_4X4_1000" ) return 3;
    if ( str_board == "DICT_5X5_50" ) return 4;
    if ( str_board == "DICT_5X5_100" ) return 5;
    if ( str_board == "DICT_5X5_250" ) return 6;
    if ( str_board == "DICT_5X5_1000" ) return 7;
    if ( str_board == "DICT_6X6_50" ) return 8;
    if ( str_board == "DICT_6X6_100" ) return 9;
    if ( str_board == "DICT_6X6_250" ) return 10;
    if ( str_board == "DICT_6X6_1000" ) return 11;
    if ( str_board == "DICT_7X7_50" ) return 12;
    if ( str_board == "DICT_7X7_100" ) return 13;
    if ( str_board == "DICT_7X7_250" ) return 14;
    if ( str_board == "DICT_7X7_1000" ) return 15;
    if ( str_board == "DICT_ARUCO_ORIGINAL" ) return 16;
    if ( str_board == "DICT_APRILTAG_16h5" ) return 17;
    if ( str_board == "DICT_APRILTAG_25h9" ) return 18;
    if ( str_board == "DICT_APRILTAG_36h10" ) return 19;
    if ( str_board == "DICT_APRILTAG_36h11" ) return 20;
    cout << " ! ! ! Error, incorrect name board ! ! ! " << endl;
    exit(0);
}
