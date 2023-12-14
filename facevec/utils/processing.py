import matplotlib.pyplot as plt
from facevec.models import *
from . import args
import tensorflow as tf
import numpy as np
import tqdm
import cv2

def imgEmoProcessing( img ):
    """
    Preprocesses the image for the emotion classifier
    Values get normalized and the image is resized to 32, 32, 1 
    returns shape = ( 1, 32, 32, 1 )
    """
    if img.shape[ -1 ] != 1:
        img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    img = cv2.resize( img, ( 32, 32 ) )
    img = np.array( img ).reshape( -1, 32, 32, 1 ) / 255
    return img

def imgPointProcessing( img ):
    '''
    Preprocesses image so that it can be inputed into the neural network

    Values are NOT normalized

    returns np.ndarray shape = ( 1, HEIGHT, WIDTH, 1 )
    '''

    img = cv2.resize( img, ( args.POINT_IMG_SIZE[ 1 ], args.POINT_IMG_SIZE[ 0 ] ) )

    img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )

    img = np.array( img ).reshape( 1, args.POINT_IMG_SIZE[ 0 ], args.POINT_IMG_SIZE[ 1 ], 1 )
    return img

def imgVecProcessing( img ):
    '''
    Preprocesses image so that it can be inputed into the neural network

    Values are normalized between 0...1

    returns np.ndarray shape = ( 1, HEIGHT, WIDTH, CHANNELS )
    '''

    img = cv2.resize( img, ( args.VEC_IMG_SIZE[ 1 ], args.VEC_IMG_SIZE[ 0 ] ) )

    if args.CHANNELS == 1:
        img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )

    img = np.array( img ).reshape( 1, args.VEC_IMG_SIZE[ 0 ], args.VEC_IMG_SIZE[ 1 ], args.CHANNELS )
    img = img / 255.
    return img

def imgDetProcessing( img ):
    img = cv2.resize( img, ( args.DET_IMG_SIZE[ 1 ], args.DET_IMG_SIZE[ 0 ] ) )

    if args.CHANNELS == 1:
        img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )

    img = np.array( img ).reshape( 1, args.DET_IMG_SIZE[ 0 ], args.DET_IMG_SIZE[ 1 ], args.CHANNELS )
    img = img / 255.
    return img

def imreadImage( fileName : str ):
    img = cv2.imread( fileName )
    return img

class KMeans():
    def __init__(self) -> None:
        """KMeans()
        Find the middlePoint of clusters
        
        Usage:
        >>> kMeans = KMeans()
        >>> kMeans.Train( np.array([[1,1]]), 10, 1 )
        >>> kMeans.GetClusterPositions()
        """
        
        self.clusterPositions = np.array([[]])

    def GetClusterPositions( self ):
        """GetClusterPositions()
        
        returns:
        self.clusterPositions
        """
        return self.clusterPositions

    def Train( self, xTrain, epochs, clusterNumber ):
        """
        Train( xTrain, epochs, clusterNumber )

        Get the distance of every cluster to every point.
        Then match every point to the nearest cluster.
        At least get mean distance to every point of a cluster and move the clustercenters.

        returns:
        None
        """
        self.clusterPositions = np.random.rand( clusterNumber, xTrain.shape[ 1 ] ) * np.max( xTrain, axis = 0, keepdims = True )
        
        for i in range( clusterNumber ):
            for j in range( xTrain.shape[ 1 ] ):
                self.clusterPositions[ i, j ] = np.random.rand( 1 ) * ( np.max( xTrain[ :, j ] ) - np.min( xTrain[ :, j ] ) ) + np.min( xTrain[ :, j ] )
        
        for i in tqdm.tqdm( range ( epochs ) ):
            distances = self.GetDistances( xTrain )
            classes   = self.GetClasses( distances, xTrain )
            self.SetMeanDistance( distances, classes ) 
            self.MoveCenters( distances )

        distances = self.GetDistances( xTrain )
        classes   = self.GetClasses( distances, xTrain )
        
        self.PPC( classes )
        self.SetMeanDistance( distances, classes )
    
    def GetDistances(self, x):
        """GetDistance( np.ndarray.shape( points, arguments ) )

        Expand x to a shape points, 1, arguments
        Expand centers to a shape 1, centers, arguments

        returns:
        difference between points and clusterpoints
        np.ndarray.shape( custernr, points, arguments )
        """
        expX = np.expand_dims( x, axis = 0 )
        expC = np.expand_dims( self.clusterPositions, axis = 1 )
        return expX - expC

    def GetClasses(self, distanceArray, xTrain):
        """GetClasses( np.ndarray.shape( points, clusters, arguments ), np.ndarray.shape( points, arguments ) )

        Get Absolute distance between clusters and points 
        Set 1 in classlist where minimum distance between cluster and a point is

        returns:
        np.ndarray.shape( points, clusternumber )
        Every point gets matched to a cluster
        """
        self.clusterDistance =  np.sum ( ( distanceArray ) ** 2, axis = -1) ** ( 1 / 2 )
        classList = np.zeros( ( xTrain.shape[ 0 ], self.clusterPositions.shape[ 0 ] ) )
        classList[ np.arange(xTrain.shape[ 0 ]), np.argmin( self.clusterDistance.T, axis = 1) ] = 1.
        return classList

    def PPC( self, classList ):
        pointsPerClass = np.zeros( ( self.clusterPositions.shape[ 0 ], 1 ) )
        for c in range( self.clusterPositions.shape[ 0 ] ):
            pointsPerClass[ c, 0 ] = np.count_nonzero( classList[ :, c ] )
        print("Points per Class")
        print( pointsPerClass )

    def SetMeanDistance( self, distanceArray, classList ):
        """SetMeanDistance( np.ndarray.shape( custernr, points, arguments ), np.ndarray.shape ( points, clusternumber ))
        
        self.meanDistance = absolute difference of every point in a cluster to the center divided through number of points in a cluster
        Set self.meanDistance
        """
        classList = np.expand_dims( classList.T, axis = -1)
        nonZeros  = np.count_nonzero( classList, axis = 1)
        distance  = np.sum ( ( distanceArray + abs ( np.expand_dims( np.min( distanceArray, axis = 1), axis = 1 ) ) ) * classList, axis = 1)
        self.meanDistance = distance / np.where( nonZeros == 0., 1., nonZeros )

    def MoveCenters( self, distanceArray):
        """MoveCenters( distanceArray, classList )

        Updates every clusterposition. For that go to minimum point in a cluster and add the mean difference to it

        Args:
            distanceArray ([np.ndarray]): [shape = ( custernr, points, arguments )]

        """
        self.clusterPositions += self.meanDistance - abs ( np.min( distanceArray, axis = 1) )
        
    def Predict( self, x ):
        """Predict( x )

        Look to which class the points belong

        Args:
            x ([np.ndarray]): [shape = ( points, arguments )]

        Returns:
            [np.ndarray]: [ shape = ( points, clusternumber ) ]
        """
        distances = self.GetDistances( x )   
        return self.GetClasses( distances, x )
    
def IOU( box1 : dict, box2 : dict ) -> float:
    """
    IOU( box1, box2 )
    
    @param box1: minimum dict{ "xmin": float, "ymin": float, "xmax": float, "ymax": float }
    @param box1: minimum dict{ "xmin": float, "ymin": float, "xmax": float, "ymax": float }
    
    computes innersection over union
    """
    xA = max( box1[ "xmin" ], box2[ "xmin" ] )
    yA = max( box1[ "ymin" ], box2[ "ymin" ] )
    xB = min( box1[ "xmax" ], box2[ "xmax" ] )
    yB = min( box1[ "ymax" ], box2[ "ymax" ] )

    interArea = max( xB - xA + 1, 0) * max( yB - yA + 1, 0)
    if interArea > 0:
        area1 = ( box1[ "xmax" ] - box1[ "xmin" ] ) * ( box1[ "ymax" ] - box1[ "ymin" ] )
        area2 = ( box2[ "xmax" ] - box2[ "xmin" ] ) * ( box2[ "ymax" ] - box2[ "ymin" ] )

        return interArea / ( area1 + area2 - interArea)
    return 0.

def NMS( boxes : list[ dict ], nmsIOU : float = 0.3 ) -> list[ dict ]:
    """
    NMS( boxes )

    @param boxes: list[ dict{ "xmin": float, "ymin": float, "xmax": float, "ymax": float } ]
    @param nmsIOU: float

    checks the iou of every box to each other and removes those with an iou over the nmsIOU
    
    returns list[ dict{ "xmin": float, "ymin": float, "xmax": float, "ymax": float } ]
    """

    settedSmall = False
    settedHuge  = False
    j = 0
    while j < len( boxes ):
        i = 0
        while i < len( boxes ):
            if i != j:
                box1 = boxes[ j ]
                box2 = boxes[ i ]

                iou = IOU( box1, box2 )
                if iou > nmsIOU:
                    if box1[ "score" ] > box2[ "score" ]:
                        boxes.remove( box2 )
                    else:
                        boxes.remove( box1 )

                    settedSmall = True
                    settedHuge  = True
                    i = 0
                    j = 0

            if not settedSmall: 
                i += 1
            else:
                settedSmall = False

        if not settedHuge:
            j += 1
        else:
            settedHuge = False

    return boxes

def GetFaceProfile( image : np.ndarray, boxes : dict[ list ], path : str, knownVectors = {} ) -> None:
    faces = boxes[ "heads" ]
    vectoriser = Vectoriser()
    pointDetector = FacePointsDetector()

    for idx, faceBox in enumerate( faces ):
        xMin = int( faceBox[ "xmin" ] ); yMin = int( faceBox[ "ymin" ] )
        xMax = int( faceBox[ "xmax" ] ); yMax = int( faceBox[ "ymax" ] )

        face = image[ yMin : yMax, xMin : xMax, : ]
        vecFace = imgVecProcessing( face )
        vec = vectoriser.vectorise( vecFace )
        attributes = vectoriser.getAttributes( vec )

        width = ( xMax - xMin ) * 0.05
        height = ( yMax - yMin ) * 0.05

        pointFace = image[ int( yMin - height ) : int( yMax + height ), int( xMin - width ) : int( xMax + width ), : ]
        pointFace = imgPointProcessing( pointFace )
        points = pointDetector.detect( pointFace )
    
        for i in range( 68 ):
            x = points[ 0, i, 0 ] / 128 * ( xMax - xMin + width * 2 )
            y = points[ 0, i, 1 ] / 128 * ( yMax - yMin + height * 2 )

            face = cv2.circle( face, ( int( x - width ), int( y - height ) ), 2, ( 0, 0, 255 ), -1 )
        
        age = attributes[ "age" ][ 0 ] / 100
        race = attributes[ "race" ]
        emo = attributes[ "emotion" ]

        fig = plt.figure( figsize = ( 10, 10 ) )
        ax1 = fig.add_subplot( 1, 2, 1 )
        ax1.imshow( face )

        gesAttributes = []
        values = []

        if len( knownVectors.keys() ) > 0:
            knownVecs = knownVectors.values()
            knownVecs = np.array( knownVecs ).reshape( -1, 1024 )
            ids = vectoriser.compare( vec, knownVecs )
            ids = np.argmax( ids, axis = -1 )
            names = knownVectors.keys()

            values.append( names[ ids ] )
            gesAttributes.append( "name" )

        gesAttributes.append( "age" )
        values.append( age )

        gesAttributes = gesAttributes + list( emo.keys() )
        gesAttributes = gesAttributes + list( race.keys() )
        gesAttributes = gesAttributes + list( attributes[ "attr" ].keys() )

        
        for key in emo.keys():
            values.append( emo[ key ][ 0 ] )

        for key in race.keys():
            values.append( race[ key ][ 0 ] )

        for key in attributes[ "attr" ].keys():
            values.append( attributes[ "attr" ][ key ][ 0 ] )

        ax2 = fig.add_subplot( 1, 2, 2 )
        ax2.barh( gesAttributes, values )

        plt.tight_layout()
        plt.savefig( path + str( idx ) + ".png" )
        plt.clf()
