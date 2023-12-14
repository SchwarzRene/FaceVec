from keras.layers import *
import tensorflow as tf

def FocalLoss( yTrue, yPred ):
    alpha = 0.25
    gamma = 2

    loss  = tf.math.reduce_sum( -yTrue[ :, :, :, 0 : 10 : 5] * tf.math.log ( tf.where ( yPred[ :, :, :, 0 : 10 : 5] == 0., 0.000001, yPred[ :, :, :, 0 : 10 : 5] ) ) )
    loss += tf.math.reduce_sum( -alpha * ( 1 - yTrue[ :, :, :, 0 : 10 : 5] ) * tf.math.log( ( 1. - ( tf.where ( yPred[ :, :, :, 0 : 10 : 5] == 1., 0.999999, tf.where( yPred[ :, :, :, 0 : 10 : 5] < 0.05, 0., yPred[ :, :, :, : 10 : 5 ] ) ) ) ) ** gamma ) )

    loss += tf.math.reduce_sum( 10 * yTrue[:, :, :, 0 : 10 : 5 ] * ( yTrue[ :, :, :, 1 : 10 : 5 ] - yPred[ :, :, :, 1 : 10 : 5 ] ) ** 2 )
 
    loss += tf.math.reduce_sum( 10 * yTrue[:, :, :, 0 : 10 : 5 ] * ( yTrue[ :, :, :, 2 : 10 : 5 ] - yPred[ :, :, :, 2 : 10 : 5 ] ) ** 2 )

    loss += tf.math.reduce_sum( 10 * yTrue[:, :, :, 0 : 10 : 5 ] * ( yTrue[ :, :, :, 3 : 10 : 5 ] - yPred[ :, :, :, 3 : 10 : 5 ] ) ** 2 )
    
    loss += tf.math.reduce_sum( 10 * yTrue[:, :, :, 0 : 10 : 5 ] * ( yTrue[ :, :, :, 4 : 10 : 5 ] - yPred[ :, :, :, 4 : 10 : 5 ] ) ** 2 )

    seenloss  = tf.math.reduce_sum( yTrue[ :, :, :, 5 ] * -yTrue[ :, :, :, -1 ] * tf.math.log ( tf.where ( yPred[ :, :, :, -1 ] == 0., 0.000001, yPred[ :, :, :, -1 ] ) ) )
    seenloss += tf.math.reduce_sum( yTrue[ :, :, :, 5 ] * -( 1 - yTrue[ :, :, :, -1 ] ) * tf.math.log( ( 1. - ( tf.where ( yPred[ :, :, :, -1 ] == 1., 0.999999, tf.where( yPred[ :, :, :, -1 ] < 0.05, 0., yPred[ :, :, :, -1  ] ) ) ) ) ) )

    loss += seenloss

    return loss / tf.cast( tf.shape( yTrue )[ 0 ] * tf.shape( yTrue )[ 1 ] * tf.shape( yTrue )[ 2 ] * tf.shape( yTrue )[ 3 ], tf.float32 )

def DropNorm( x ):
    x = LeakyReLU( alpha = 0.1 )( x )
    x = BatchNormalization()( x )
    x = Dropout( 0.05 )( x )
    return x

def ResidualBlock( inp, id, filters, addLayer = True, av = False ):
    if av:
        inp = AveragePooling2D( ( 3, 3 ), strides = ( 2, 2 ), padding = "same" )( inp )

    if inp.shape[ -1 ] != filters:
        inp = Conv2D( filters, ( 1, 1 ), use_bias = True )( inp )
        inp = DropNorm( inp )

    conv1 = Conv2D( filters, ( 3, 3 ), padding = "same", use_bias = True )( inp )
    conv1 = DropNorm( conv1 )

    conv2 = Conv2D( filters, ( 1, 1 ), padding = "same", use_bias = True )( conv1 )
    conv2 = DropNorm( conv2 )
    
    conv3 = Conv2D( filters, ( 3, 3 ), padding = "same", use_bias = True )( conv2 )
    conv3 = DropNorm( conv3 )

    if addLayer:
        add1 = add([ inp, conv3 ])
        add1 = DropNorm( add1 )
        return add1
    else:
        return DropNorm( conv3 )
    
def MiddleResBlock( inp, id, filters, addLayer = True, av = False ):
    if av:
        inp = AveragePooling2D( ( 3, 3 ), strides = ( 2, 2 ), padding = "same" )( inp )

    if inp.shape[ -1 ] != filters:
        inp = Conv2D( filters, ( 1, 1 ), use_bias = True )( inp )
        inp = DropNorm( inp )

    conv1 = Conv2D( filters, ( 3, 3 ), padding = "same", use_bias = True )( inp )
    conv1 = DropNorm( conv1 )

    conv2 = Conv2D( filters, ( 5, 5 ), padding = "same", use_bias = True )( conv1 )
    conv2 = DropNorm( conv2 )

    conv3 = Conv2D( filters, ( 3, 3 ), padding = "same", use_bias = True )( conv2 )
    conv3 = DropNorm( conv3 )

    if addLayer:
        add1 = add([ inp, conv3 ] )
        add1 = DropNorm( add1 )
        return add1
    else:
        return DropNorm( conv3 )

def LeakyDenseBlock( inp, denseUnits = 128, outputUnits = 2, activation = "sigmoid" ):
    dense = Dense( denseUnits, use_bias = True )( inp )
    dense = LeakyReLU( alpha = 0.1 )( dense )
    out   = Dense( outputUnits, activation = "sigmoid", use_bias = True )( dense )
    return out

def Head( inp ):
    out = []

    xyOut = LeakyDenseBlock( inp )
    whOut = LeakyDenseBlock( inp, activation = "tanh" )
    scoreOut = LeakyDenseBlock( inp, outputUnits = 1 )
    
    out.append( scoreOut )
    out.append( xyOut )
    out.append( whOut )
    
    xyOut = LeakyDenseBlock( inp )
    whOut = LeakyDenseBlock( inp, activation = "tanh" )
    scoreOut = LeakyDenseBlock( inp, outputUnits = 1 )
    seenOut = LeakyDenseBlock( inp, outputUnits = 1 )
    
    out.append( scoreOut )
    out.append( xyOut )
    out.append( whOut )
    out.append( seenOut )

    return concatenate( out )

def SetupNetwork():
    inp = Input( shape = ( None, None, 3 ) ) #256 | 512

    #x = MiddleResBlock( inp, 1111, 16, av = False )
    conv1 = Conv2D( 16, ( 5, 5 ), strides = ( 2, 2 ), padding = "same" )( inp )
    conv1 = DropNorm( conv1 )

    x10 = ResidualBlock( conv1, 1, 32 ) #128 | 256

    x10 = ResidualBlock( x10, 3, 32, av = True )   #64 | 128
    x11 = MiddleResBlock( x10, 4, 32 )
    #x11 = ResidualBlock( x11, 44, 32 )
    x = add( [ x10, x11 ] )

    x10 = ResidualBlock( x, 5, 64, av = True )  #32 | 64
    x11 = ResidualBlock( x10, 66, 64 )
    x = add( [ x10, x11 ] )

    x10 = ResidualBlock( x, 9, 128, av = True ) #16 | 32
    x11 = ResidualBlock( x10, 100, 128 )

    x1 = add( [ x10, x11 ] )

    x10 = ResidualBlock( x1, 11, 128, av = True ) #8 | 16
    x11 = ResidualBlock( x10, 12, 128 )

    x2 = add( [ x10, x11 ] )

    x10 = ResidualBlock( x2, 13, 128, av = True ) #4 | 8
    x11 = ResidualBlock( x10, 14, 128 )

    x3 = add( [ x10, x11 ] )

    up1 = UpSampling2D( ( 2, 2 ) )( x3 )

    x10 = ResidualBlock( up1, 15, 128 ) #4 | 8
    x11 = ResidualBlock( x10, 16, 128 )

    x2 = add( [ x10, x11, x2 ] )

    up1 = UpSampling2D( ( 2, 2 ) )( x2 )
    
    x10 = ResidualBlock( up1, 17, 128 ) #4 | 8
    x11 = ResidualBlock( x10, 18, 128 )

    x1 = add( [ x10, x11, x1 ] )

    x1 = Head( x1 )
    x2 = Head( x2 )
    x3 = Head( x3 )

    model = tf.keras.models.Model( inp, [ x1, x2, x3 ] )
    model.compile( optimizer = tf.keras.optimizers.SGD(), loss = FocalLoss )
    return model
