# Community modules
from pandas import DataFrame
from numpy  import zeros, corrcoef, arange, concatenate
from sklearn.feature_selection import mutual_info_regression as MI
from pyEDM import Embed

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def MI_tau( args, data ):
    '''Mutual Information decay vs time-lag tau on columns to target.

    Note that sklearn.feature_selection.mutual_info_regression uses a
    k-nearest neighbors MI algorithm from:
      Estimating mutual information, PHYSICAL REVIEW E 69, 066138 (2004)
      Kraskov, Stögbauer, Grassberger.
        Mutual entropy estimators parametrized by an integer k > 1
        using kth neighbor distance statistics in the joint space.
        Choosing small k reduces general systematic errors, while
        large k leads to smaller statistical errors. The choice of the
        particular estimator depends thus on the size of the data
        sample and on whether bias or variance is to be minimized.
    '''

    offset  = abs( args.tau ) * ( args.maxLag - 1 )
    lags    = arange( 1, args.maxLag + 1 )
    numVars = len( args.columns )
    numRows = len( lags )

    target = data[ args.target ].values
    # Remove leading (tau < 0) or trailing (tau > 0) to match Embed NaN
    if args.tau < 0 :
        target = target[ offset : data.shape[0] ]
    else :
        target = target[ 0 : data.shape[0] - offset ]
    y = target

    # Matrix to hold MI & CC values
    M = zeros( (numRows, numVars) )
    C = zeros( (numRows, numVars) )

    colNamesMI = [] # Build column names for output DataFrame
    colNamesCC = []

    # MI all variables
    for col_i in range( numVars ):
        column  = args.columns[ col_i ]
        colName = column + ":" + args.target
        colNamesMI.append( 'MI[' + colName + ']' )
        colNamesCC.append( 'CC[' + colName + ']' )

        # maxLag dimensional embedding
        E = Embed( dataFrame = data, E = args.maxLag,
                   tau = args.tau, columns = column )

        for row_i in range( args.maxLag ) :

            # Remove leading (tau < 0) or trailing (tau > 0) NaN
            if args.tau < 0 :
                x = E.iloc[ offset : data.shape[0], row_i ].values
            else :
                x = E.iloc[ 0 : data.shape[0] - offset, row_i ].values

            # Linear
            CC = corrcoef( x, y )[0,1]

            # Mutual information
            IXY = MI( x.reshape(-1,1), y,
                      discrete_features = False,
                      n_neighbors = args.MI_neighbors,
                      copy = True, random_state = None )[0]

            M[ row_i, col_i ] = round( IXY, 5 )
            C[ row_i, col_i ] = round( CC,  5 )

    # Output DataFrame
    MC   = concatenate( (lags.reshape(-1,1), M, C), axis = 1 )
    MCdf = DataFrame( MC )
    MCdf.columns = ["lag"] + colNamesMI + colNamesCC

    #if args.outFile :
    #    MCdf.to_csv( args.outFile )

    if args.verbose :
        print( MCdf )

    return MCdf
