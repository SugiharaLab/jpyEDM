# Python distribution modules
from math import sqrt, ceil

# Community modules
import matplotlib.pyplot    as     plt
from   matplotlib.pyplot    import axhline
from   mpl_toolkits.mplot3d import axes3d
from   numpy                import arange
from   pyEDM                import ComputeError

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def PlotEmbedDimension( df, args ):
    title = args.inputFile + "\nTp=" + str(args.Tp)

    ax = df.plot( 'E', 'rho', title = title, linewidth = 3 )
    ax.set( xlabel = "Embedding Dimension",
            ylabel = "Prediction Skill ρ" )

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def PlotPredictNonlinear( df, args ):

    if args.embedded :
        E = len( args.columns )
    else :
        E = args.E

    title = args.inputFile + "\nE=" + str( E )

    ax = df.plot( 'Theta', 'rho', title = title, linewidth = 3 )
    ax.set( xlabel = "S-map Localisation (θ)",
            ylabel = "Prediction Skill ρ" )

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def PlotPredictInterval( df, args ):

    if args.embedded :
        E = len( args.columns )
    else :
        E = args.E

    title = args.inputFile + "\nE=" + str( E )

    ax = df.plot( 'Tp', 'rho', title = title, linewidth = 3 )
    ax.set( xlabel = "Forecast Interval",
            ylabel = "Prediction Skill ρ" )

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def Plot3D( D, columnList ):

    fig = plt.figure()
    ax  = fig.add_subplot( projection = '3d' )

    X, Y, Z = D[ columnList[0] ], \
              D[ columnList[1] ], \
              D[ columnList[2] ]

    # Plot
    ax.scatter( X, Y, Z, zdir = 'z', s = 20, c = None, depthshade = True )
    ax.set_xlabel( columnList[0] )
    ax.set_ylabel( columnList[1] )
    ax.set_zlabel( columnList[2] )

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def PlotCCM( libMeans, args ):
    title = args.inputFile + "\nE=" + str( args.E )

    ax = libMeans.plot( 'LibSize',
                        [ libMeans.columns[1], libMeans.columns[2] ],
                        title = title, linewidth = 3 )
    ax.set( xlabel = "Library Size", ylabel = "Cross Map ρ" )
    axhline( y = 0, linewidth = 1 )

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def PlotObsPred_( df, args ):

    # stats: {'MAE': 0., 'RMSE': 0., 'rho': 0. }
    stats = ComputeError( df['Observations'], df['Predictions' ] )

    if args.embedded :
        E = len( args.columns )
    else :
        E = args.E

    title = args.inputFile + "\nE=" + str( E ) + " Tp=" + str( args.Tp ) +\
            "  ρ="   + str( round( stats['rho'],  2 ) ) +\
            " RMSE=" + str( round( stats['RMSE'], 2 ) )

    df.plot( df.columns[0], ['Observations', 'Predictions'],
             title = title, linewidth = 3 )

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def PlotCoeff_( df, args ):

    if args.embedded :
        E = len( args.columns )
    else :
        E = args.E

    title = args.inputFile + "\nE=" + str( E ) + " Tp=" + str( args.Tp ) +\
            "  S-Map Coefficients"

    time_col = df.columns[0]
    # Coefficient columns can be in any column
    coef_cols = [ x for x in df.columns if time_col not in x ]

    df.plot( time_col, coef_cols, title = title, linewidth = 3,
             subplots = True )

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------
def PlotMutualInfo( df, args ):

    colNames = df.columns # lag, MI[v1:v2]..., CC[v1:v2]
    lags     = df.loc[ :,'lag' ]

    numVars  = len( args.columns )
    nRowCol  = ceil( sqrt( numVars ) )

    # If numVars > 1, create a square matrix subplot
    fig, axs = plt.subplots( nRowCol, nRowCol )

    title = 'Mutual Info & Correlation  nn:' + str( args.MI_neighbors )
    fig.suptitle( title )

    if numVars == 1 :
        axs.plot( lags, df.iloc[ :, 1 ], label = 'MI' )
        axs.plot( lags, df.iloc[ :, 2 ], label = 'CC' )
        axs.legend()
        axs.set_title( args.columns[ 0 ] + ":" + args.target, y = 0 )
    else:
        MI_cols = range( 1, numVars + 1 )
        CC_cols = range( numVars + 1, 2 * numVars + 1 )
        k = 0
        for i in range( nRowCol ) :
            if k >= numVars :
                break

            for j in range( nRowCol ) :
                ax = axs[i,j]
                ax.plot( lags, df.loc[ :, colNames[MI_cols[k]] ], label='MI' )
                ax.plot( lags, df.loc[ :, colNames[CC_cols[k]] ], label='CC' )
                ax.legend()
                ax.set_title( args.columns[ k ] + ":" + args.target,y=0 )

                k = k + 1

                if k >= numVars :
                    break
