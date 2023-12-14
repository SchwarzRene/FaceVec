import matplotlib.pyplot as plt
import facevec.utils.processing
import facevec.utils.layers
import tensorflow as tf
import onnxruntime
import numpy as np
import pickle
import cv2
import os


class Vectoriser():
    def __init__( self ):
        path = __file__.replace( "\\", "/" )
        path = path.replace( "models.py", "utils/Models/" )

        try:
            self.__gpuFaceVec = tf.keras.models.load_model( path + "Network/" )
        except Exception as e:
            print( "Error loading Vectoriser Model" )
            print( e )

        try:
            self.__faceVec = onnxruntime.InferenceSession( path + "Network/network.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider" ] )
        except Exception as e:
            print( "Error loading Vectoriser onnx Model" )
            print( e )

        try:
            self.__emotionModel = tf.keras.models.load_model( path + "Emotion/" )
        except Exception as e:
            print( "Error loading Emotion Model" )
            print( e )

        try:
            self.__faceAtt = tf.keras.models.load_model( path + "AttModel/" )
        except Exception as e:
            print( "Error loading Attribute Model" )
            print( e )

        try:
            self.__faceAge = tf.keras.models.load_model( path + "AgeModel/" )
        except Exception as e:
            print( "Error loading Age Model")
            print( e )

        try:
            self.__clusters = pickle.load( open( path + "frontBack.pickle", "rb" ) )
        except Exception as e:
            print( "Error loading cluster Positions" )
            print( e )

        try:
            self.__pointer = tf.keras.models.load_model( path + "Pointer/" )
        except Exception as e:
            print( "Error loading face point model" )
            print( e )


        attributeList = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
        self.__faceAttributes = attributeList.split( " " )

        self.__races = [ "white", "black", "asian", "indian", "other" ]

        self.__emotions = [ "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise" ]

    def compare( self, a : np.ndarray, b : np.ndarray ):
        self.__checkShape( a )
        self.__checkShape( b )
        
        return np.sum( ( a - b ) ** 2, axis = -1, keepdims = True )
     
    def __checkShape( self, vec ):
        if vec.shape[ -1 ] != 1024:
            raise ValueError( "Error dimension shape of face 1 must be ( -1, 1024 )" )
    
    def getAttributes( self, face : np.ndarray ):
        """
        Get Face Attributes
        
        @param face: np.ndarray.shape( -1, 1024 )

        face Points: dict( point: { "x": [ batch ], "y": [ batch ] ) )
        face Attributes: dict( attr: [ batch, [ 0 ... 1 ] ] )
        face Age: np.ndarray( -1, 1 )
        face front: np.ndarray( -1, 1 ) 1 if face is front else 0
        face emotion: dict( emotion: [ batch, [ 0 ... 1 ] ] )

        returns { "points": face Points, "attr": face Attributes, "age": face Age, "emotion": face emotion }
        """
        self.__checkShape( face )

        dist = self.compare( np.expand_dims( self.__clusters, axis = 0 ), np.expand_dims( face, axis = 1 ) )
        dist = np.squeeze( dist, axis = -1 )
        dist = np.argmin( dist, axis = -1 )
        dist = np.where( ( dist == 1 ) | ( dist == 2 ), 1, 0 )

        predAtt = self.__faceAtt( face, training = False )
        predAge = self.__faceAge( face, training = False )

        attrDict = {}
        for idx, key in enumerate( self.__faceAttributes ):
            attrDict.update( { key: predAtt[ :, idx ] } )
 
        race = predAge[ 1 ]

        #Return index of maximum softmax value
        predAge = np.sum( predAge[ 0 ] * np.expand_dims( np.arange( 1, 117, 1 ), axis = 0 ), axis = -1 )

        raceDict = {}
        for i, r in enumerate( self.__races ):
            raceDict.update( { r: race[ :, i ] } )
            
        predEmo = self.__emotionModel( face, training = False )
        emotionDict = {}
        for idx, key in enumerate( self.__emotions ):
            emotionDict.update( { key: predEmo[ :, idx ] } )

        return { "attr": attrDict, "age": predAge, "front": dist, "race": raceDict, "emotion": emotionDict }

    def vectorise( self, face : np.ndarray, gpu = False ):
        """
        @param face: np.ndarray( -1, 64, 64, 3 ) [ 0 ... 1 ]
        """
        if gpu:
            return self.__gpuFaceVec( face, training = False )
        else:
            face = face.astype( np.float32 )
            return self.__faceVec.run( [ "batch_normalization_69" ], { "input_1": face } )[ 0 ]
        
    
class PseudoGenerator():
    def __init__( self ):
        path = __file__.replace( "\\", "/" )
        path = path.replace( "models.py", "utils/Models/Detector/GeneratorData.pickle" )
        data = pickle.load( open( path, "rb" ) )
        self.anchorList = data[ "Anchors" ]
        self.classList  = data[ "Classes" ]
        self.scaleFactors  = data[ "ScaleFactors" ]

class Detector():
    def __init__( self ):
        path = __file__.replace( "\\", "/" )
        path = path.replace( "models.py", "utils/Models/" )

        self.__model = onnxruntime.InferenceSession( path + "Detector/model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider" ] )
        self.__gpuModel = facevec.utils.layers.SetupNetwork()
        self.__gpuModel.load_weights( path + "Detector/model.hdf5" )

        self.__generator = PseudoGenerator()

        newAnchorList = [] 
        for i in range( 3 ):
            for key in list( self.__generator.anchorList.keys() ):
                newAnchorList.append( self.__generator.anchorList[ key ][ i ] )

        self.__anchorList = np.reshape( newAnchorList, ( 3, len( self.__generator.classList ), 2 ) )

    def detect( self, image : np.ndarray, threashold = { "bodyThreashold": 0.2, "headThreashold": 0.3 }, gpu = False ) -> dict[ list[ dict ] ]:
        """
        detect( image : np.ndarray, threashold = { "bodyThreashold": 0.2, "headThreashold": 0.3 } ) -> list[ dict ]

        @param image: np.ndarray ideal shape = ( -1, 512, 512, 3 )
        @param threashold: { "bodyThreashold": 0.2, "headThreashold": 0.3 }

        Detects bodies and faces in an image
        Works on multiple images

        returns 
        [ { "heads": [ { "score": float, "xmin": float, "ymin": float, "xmax": float, "ymax": float, "classe": "head", "seen": float } ], "bodies": [ { "score": float, "xmin": float, "ymin": float, "xmax": float, "ymax": float, "classe": "head", "seen": float } ] } ]
        """
        image = image.astype( np.float32 )

        if not gpu:
            pred = self.__model.run( [ "concatenate", "concatenate_1", "concatenate_2" ], { "input_1": image } )
        else:
            pred = self.__gpuModel.predict( image )

        imgWidth  = image.shape[ 2 ]
        imgHeight = image.shape[ 1 ]      

        output = []
        for batchSize in range( image.shape[ 0 ] ):
            gesHeads = []
            gesBodies = []
            for j in range( 3 ):
                grid  = np.array( pred[ j ] )

                stridesX = imgWidth // ( 16 * 2 ** j )
                stridesY = imgHeight // ( 16 * 2 ** j )

                posX = np.expand_dims( np.ones( ( stridesY, 1 ) ) * np.arange( stridesX ), 2 )
                posY = np.ones( ( stridesX, 1 ) ) * np.arange( stridesY )
                posY = np.expand_dims( posY.T, 2 )
                
                grid[ batchSize, :, :, 1 : : 5 ] = imgWidth  / stridesX * ( posX + grid[ batchSize, :, :, 1 : : 5 ] )
                grid[ batchSize, :, :, 2 : : 5 ] = imgHeight / stridesY * ( posY + grid[ batchSize, :, :, 2 : : 5 ] )
                
                grid[ batchSize, :, :, 3 : : 5 ] = ( grid[ batchSize, :, :, 3 : : 5 ] * self.__generator.scaleFactors[ j ][ 0 ] + 1 ) * self.__anchorList[ j, :, 0 ]
                grid[ batchSize, :, :, 4 : : 5 ] = ( grid[ batchSize, :, :, 4 : : 5 ] * self.__generator.scaleFactors[ j ][ 1 ] + 1 ) * self.__anchorList[ j, :, 1 ]

                ###get bodys
                indexes = np.argwhere( grid[ batchSize, :, :, 0 ] > threashold[ "bodyThreashold" ] )

                positionIndexes = indexes[ :, 0 ] * grid.shape[ 2 ]  + indexes[ :, 1 ]
                
                positionIndexes = positionIndexes.astype( np.int16 )
                
                flattendGrid = np.reshape( grid[ batchSize, :, :, 0 : 5 : ], ( -1, 5 ) )
                boxes = flattendGrid[ positionIndexes, : ]
                
                boxes[ :, 1 ] = boxes[ :, 1 ] - boxes[ :, 3 ] / 2 
                boxes[ :, 3 ] = boxes[ :, 1 ] + boxes[ :, 3 ] 

                boxes[ :, 2 ] = boxes[ :, 2 ] - boxes[ :, 4 ] / 2 
                boxes[ :, 4 ] = boxes[ :, 2 ] + boxes[ :, 4 ] 

                boxes = np.concatenate( [ boxes, np.zeros( ( len( boxes ), 1 ), dtype = np.int32 ) ], axis = -1 )
    
                if j == 0: 
                    gesBodies = boxes 
                else:
                    gesBodies   = np.concatenate( [ gesBodies, boxes ] )

                ###get heads
                indexes = np.argwhere( grid[ batchSize, :, :, 5 ] > threashold[ "headThreashold" ] )

                positionIndexes = indexes[ :, 0 ] * grid.shape[ 2 ]  + indexes[ :, 1 ]
                
                positionIndexes = positionIndexes.astype( np.int16 )
                
                flattendGrid = np.reshape( grid[ batchSize, :, :, 5 : : ], ( -1, 6 ) )
                boxes = flattendGrid[ positionIndexes, : ]
                
                boxes[ :, 1 ] = boxes[ :, 1 ] - boxes[ :, 3 ] / 2 
                boxes[ :, 3 ] = boxes[ :, 1 ] + boxes[ :, 3 ] 

                boxes[ :, 2 ] = boxes[ :, 2 ] - boxes[ :, 4 ] / 2 
                boxes[ :, 4 ] = boxes[ :, 2 ] + boxes[ :, 4 ] 

                if j == 0:
                    gesHeads = boxes
                else:
                    gesHeads = np.concatenate( [ gesHeads, boxes ] )

            gesHeads = list( map( lambda x: { "score": x[ 0 ], "xmin": int( x[ 1 ] ), "ymin": int( x[ 2 ] ), "xmax": int( x[ 3 ] ), "ymax": int( x[ 4 ] ), "seen": x[ 5 ], "classe": "head" }, gesHeads ) )
            gesBodies = list( map( lambda x: { "score": x[ 0 ], "xmin": int( x[ 1 ] ), "ymin": int( x[ 2 ] ), "xmax": int( x[ 3 ] ), "ymax": int( x[ 4 ] ), "classe": "body" }, gesBodies ) )

            gesHeads = facevec.utils.processing.NMS( gesHeads )
            gesBodies = facevec.utils.processing.NMS( gesBodies )
            output.append( { "heads": gesHeads, "bodies": gesBodies } )

        return output
    
    def saveBoxes( self, image : np.ndarray, boxes : dict[ list ], path : str ) -> None:
        """
        saveBoxes( image, boxes, path )

        @param image: np.ndarray -> the image where the boxes are detected
        @param boxes: dict[ list ] -> the detected boxes in the image
        @param path: str -> the path where the boxes should be saved

        Save all detected boxes in the image

        return None
        """
        counter = 0
        for key in boxes.keys():
            for box in boxes[ key ]:
                xMin = box[ "xmin" ]; xMax = box[ "xmax" ]; yMin = box[ "ymin" ]; yMax = box[ "ymax" ]
                imgBox = image[ int( yMin ) : int( yMax ), int( xMin ) : int( xMax ), : ]
                cv2.imwrite( path + str( counter ) + ".png", imgBox )

def PixelLoss( yTrue, yPred ):
    return 0

class FacePointsDetector():
    def __init__( self ) -> None:
        """
        Initialise FPDetector

        Load keras model

        :params

        returns None
        """
        path = __file__.replace( "\\", "/" )
        path = path.replace( "models.py", "utils/Models/" )

        self.model = tf.keras.models.load_model( path + "FPDetector.h5", custom_objects = { "PixelLoss": PixelLoss } )

    def detect( self, x ) -> np.ndarray:
        """
        FPDetector( x )
        :param x = np.ndarray( -1, 128, 128, 1 ), NOT normalized, values 0-255

        Predicts 68 face points in a 3 dimensional space
        Points are not normalized

        >>> x-Cords = points[ :, :, 0 ], origin = image( 0, 0 )
        >>> y-Cords = points[ :, :, 1 ], origin = image( 0, 0 )
        >>> z-Cords = points[ :, :, 2 ], origin = nose top, points[ :, 27, 2 ]

        returns np.ndarray( -1, 68, 3 )
        """
        #Predict normalized points
        predPoints = self.model( x, training = False )

        #Bring points to original size
        points = np.concatenate( [ np.reshape( predPoints[ 0 ][ :, : ], (-1, 68, 1 ) ), np.reshape( predPoints[ 1 ][ :, : ], (-1, 68, 1 ) ), np.reshape( predPoints[ 2 ][ :, : ], (-1, 68, 1 ) ) ], axis = -1)

        points[ :, :, 0 ] = points[ :, :, 0 ] * ( 32.30667388375604 + 1e-10 ) * 2 + 64.0
        points[ :, :, 1 ] = points[ :, :, 1 ] * ( 21.87913491694697 + 1e-10 ) * 2 + 78.84957773881295
        points[ :, :, 2 ] = points[ :, :, 2 ] * ( 13.158591786482512 + 1e-10 ) * 5.5 + 2.850532379623897
        points = np.reshape( points, ( -1, 68, 3 ) )

        return points 
        
    def FaceSimilarity( self, points1, points2 ) -> np.ndarray:
        """
        FaceSimilarity( points1, points2 )

        :param points1 = np.ndarray( -1, 68, 3 ), predicted points of __call__
        :param points2 = np.ndarray( -1, 68, 3 ), predicted points of __call__

        returns: np.ndarray( -1, 1 ), smaller is better
        """
        #Get distance from origin to every point
        distance1 = np.reshape ( np.sum( ( points1[ :, :, : ] - np.expand_dims ( points1[ :, 27, : ], axis = 1 ) ) ** 2, axis = -1 ) ** 0.5 , ( -1, 68 ) )
        distance2 = np.reshape ( np.sum( ( points2[ :, :, : ] - np.expand_dims ( points2[ :, 27, : ], axis = 1 ) ) ** 2, axis = -1 ) ** 0.5 , ( -1, 68 ) )

        #Returns absolute distance between the two face points
        return np.linalg.norm( distance1 - distance2, axis = 1 ) 
    
    def Plot3DFaceMask( self, points ) -> None:
        """
        Plot3DFaceMask( points )

        :param poins = np.ndarray( -1, 68, 3 ), predicted points of __call__

        Creates a 3d matplotlib scatter plot

        returns: None
        """

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter( points[ :, :, 0 ],  points[ :, :, 1 ],  points[ :, :, 2 ] )
        plt.show()

    def Plot2DFaceMask( self, points, image ) -> np.ndarray: 
        """
        Plot2DFaceMask( points, image )

        :param points = np.ndarray( -1, 68, 3 ), predicted points of __call__
        :param image  = np.ndarray( 128, 128, 1 ) or np.ndarray( 128, 128, 3 )

        Creates a circle for every point on the image

        returns np.ndarray( 128, 128, 3 ) or np.ndarray( 128, 128, 3 )
        """
        color = 1 if np.max( image ) <= 1 else 255
        for i in range( 68 ):
            image = cv2.circle(image, ( int ( points[ 0, i, 0 ] ), int ( points[ 0, i, 1 ] ) ), 2, ( color, 0, 0), -1)
        return image
    
    def Create3DFace( self, points, image ) -> None:
        fig = plt.figure()
        ax = fig.add_subplot( projection = '3d' )
        ax.scatter( points[ :, :, 0 ],  points[ :, :, 1 ],  points[ :, :, 2 ], marker = 's' )
        plt.show()

class EmotionClassifier():
    def __init__( self ):
        path = __file__.replace( "\\", "/" )
        path = path.replace( "models.py", "utils/Models/" )

        self.__model = tf.keras.models.load_model( path + "Emotion_32" )
        self.__emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    def detect( self, face : np.ndarray ) -> dict:
        """
        detect( face )

        @param face: np.ndarray with shape ( -1, 32, 32, 1 ), face is normalized with values between [ 0...1 ]
        @param emotion: dict[ emotion: [ batchSize, ] ]

        This model can detect ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        The image size is 32, 32, 1 to reduce computational complexity
        """

        pred = self.__model( face, training = False )
        pred = np.array( pred ).reshape( -1, len( self.__emotions ) )

        emotionDict = {}
        for idx, emotion in enumerate( self.__emotions ):
            emotionDict.update( { emotion: pred[ :, idx ] } )

        return emotionDict

if __name__ == "__main__":
    vect = Vectoriser()