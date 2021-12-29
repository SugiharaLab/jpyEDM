import matplotlib.pyplot    as     plt
from   matplotlib.pyplot    import axhline
from   mpl_toolkits.mplot3d import axes3d
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
    title = args.inputFile + "\nE=" + str(args.E)
    
    ax = df.plot( 'Theta', 'rho', title = title, linewidth = 3 )
    ax.set( xlabel = "S-map Localisation (θ)",
            ylabel = "Prediction Skill ρ" )

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def PlotPredictInterval( df, args ):
    title = args.inputFile + "\nE=" + str(args.E)
    
    ax = df.plot( 'Tp', 'rho', title = title, linewidth = 3 )
    ax.set( xlabel = "Forecast Interval",
            ylabel = "Prediction Skill ρ" )

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def Plot3D( D, columnList ):

    fig = plt.figure().gca( projection = '3d' )

    X, Y, Z = D[ columnList[0] ], \
              D[ columnList[1] ], \
              D[ columnList[2] ]

    # Plot
    fig.scatter( X, Y, Z, zdir = 'z', s = 20, c = None, depthshade = True )
    fig.set_xlabel( columnList[0] )
    fig.set_ylabel( columnList[1] )
    fig.set_zlabel( columnList[2] )

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def PlotCCM( libMeans, args ):
    title = args.inputFile + "\nE=" + str(args.E)

    ax = libMeans.plot( 'LibSize',
                        [ libMeans.columns[1], libMeans.columns[2] ],
                        title = title, linewidth = 3 )
    ax.set( xlabel = "Library Size", ylabel = "Correlation ρ" )
    axhline( y = 0, linewidth = 1 )

#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def PlotObsPred_( df, args ):
    
    # stats: {'MAE': 0., 'RMSE': 0., 'rho': 0. }
    stats = ComputeError( df['Observations'], df['Predictions' ] )

    title = args.inputFile + "\nE=" + str(args.E) + " Tp=" + str(args.Tp) +\
            "  ρ="   + str( round( stats['rho'],  2 ) )   +\
            " RMSE=" + str( round( stats['RMSE'], 2 ) )

    df.plot( df.columns[0], ['Observations', 'Predictions'],
             title = title, linewidth = 3 )
    
#----------------------------------------------------------------------------
# 
#----------------------------------------------------------------------------
def PlotCoeff_( df, args ):
                     
    title = args.inputFile + "\nE=" + str(args.E) + " Tp=" + str(args.Tp) +\
            "  S-Map Coefficients"

    time_col = df.columns[0]
    # Coefficient columns can be in any column
    coef_cols = [ x for x in df.columns if time_col not in x ]

    df.plot( time_col, coef_cols, title = title, linewidth = 3,
             subplots = True )
