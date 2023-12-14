import cv2
import os
import facevec

files = os.listdir( "./testimages/" )

det = facevec.models.Detector()

for file in files[ 1 : ]:
    img = facevec.utils.processing.imreadImage( "./testimages/" + file )
    img = cv2.resize( img, ( 512, 512 ) )
    detImg = facevec.utils.processing.imgDetProcessing( img )
    boxes = det.detect( detImg )
    for idx, box in enumerate( boxes[ "heads" ] ):
        face = img[ box[ "ymin" ] : box[ "ymax" ], box[ "xmin" ] : box[ "xmax" ], : ]
        cv2.imwrite( "./testimages/extracted/" + file.split( "." )[ 0 ] + "_" + str( idx ) + ".png", face )