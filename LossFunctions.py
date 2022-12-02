def absolute_error(y_in,y_out):
    """Returns absolute error"""

    mae = tf.keras.losses.MeanAbsoluteError()
    d1 = tf.abs(mae([y_in[0],y_in[1]], [y_out[0],y_out[1]]))
    d2 = tf.abs(mae([y_in[0],y_in[1]], [y_out[2],y_out[3]]))
    
    return tf.math.minimum(d1,d2)

def percentage_error(y_in,y_out):
    """Returns percentage error"""

    mae = tf.keras.losses.MeanPercentageError()
    d1 = tf.abs(mae([y_in[0],y_in[1]], [y_out[0],y_out[1]]))
    d2 = tf.abs(mae([y_in[0],y_in[1]], [y_out[2],y_out[3]]))
    
    return tf.math.minimum(d1,d2)

def absolute_plus_error_(y_in,y_out):
    """Returns absolute error with a slight adjustment to remove NaN values"""

    mae = tf.keras.losses.MeanAbsoluteError()
    d1 = tf.abs(mae([y_in[0],y_in[1]], [y_out[0],y_out[1]])) + 0.005
    d2 = tf.abs(mae([y_in[0],y_in[1]], [y_out[2],y_out[3]])) + 0.005

    return tf.math.minimum(d1,d2)
