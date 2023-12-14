import facevec.models
import facevec.utils.processing
import numpy as np
import cv2

det = facevec.models.Detector()
vec = facevec.models.Vectoriser()
emo = facevec.models.EmotionClassifier()

cap = cv2.VideoCapture( 0 )

counter = 3000
while True:
    showImg = np.zeros( ( 512, 1024, 3 ), dtype = np.uint8 )
    ret, frame = cap.read()
    frame = cv2.resize( frame, ( 512, 512 ) )
    detFrame = facevec.utils.processing.imgDetProcessing( frame )
    boxes = det.detect( detFrame )

    for box in boxes[ "heads" ]:
        box[ "xmin" ] = int( box[ "xmin" ] ); box[ "xmax" ] = int( box[ "xmax"] ) 
        box[ "ymin" ] = int( box[ "ymin" ] ); box[ "ymax" ] = int( box[ "ymax" ] )

        face = frame[ box[ "ymin" ] : box[ "ymax" ], box[ "xmin" ] : box[ "xmax" ], : ]
        cv2.imwrite( "./testimages/" + str( counter ) + ".jpg", face )

        frame = cv2.rectangle( frame, ( box[ "xmin" ], box[ "ymin" ] ), ( box[ "xmax" ], box[ "ymax" ] ), ( 0,0,255 ), 3 )
        counter += 1

        emoFace = cv2.resize( face, ( 64, 64 ) )
        emoFace = cv2.cvtColor( emoFace, cv2.COLOR_RGB2GRAY )
        emoFace = np.array( emoFace ).reshape( -1, 64, 64, 1 ) / 255.
        emotions = emo.detect( emoFace )
        for i, e in enumerate( emotions.items() ):
            width = 512 // len( list( emotions.items() ) )
            cv2.rectangle( showImg, ( 512, width * i ), ( 512 + int( e[ 1 ] * 512 ), width * ( i + 1 ) ), ( 255, 0, 0 ), -1 )
            cv2.putText( showImg, e[ 0 ], ( 535, width * ( i + 1 ) - width // 2 ), cv2.FONT_HERSHEY_COMPLEX, 1, ( 0, 255, 0 ), 2 )
        
    showImg[ 0 : 512, 0 : 512, : ] = frame

    cv2.imshow( "", showImg )
    cv2.waitKey( 1 )