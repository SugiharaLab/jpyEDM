
from argparse import ArgumentParser

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
def ParseCmdLine():
    '''
    nargs = '*' All command-line args are gathered into a list.
    nargs = '+' All command-line args are gathered into a list, and,
                an error message generated if not at least one argument.
    '''
    
    parser = ArgumentParser( description = 'jpyEDM' )
    
    parser.add_argument('-m', '--method',
                        dest   = 'method', type = str, 
                        action = 'store', default = 'Simplex',
                        help = 'Type of projection Simplex or SMap.')

    parser.add_argument('-slv', '--solver',
                        dest   = 'solver', type = str, 
                        action = 'store', default = None,
                        help = 'sklearn solver for SMap.')

    parser.add_argument('-p', '--pred', nargs = '+',
                        dest   = 'pred', type = int, 
                        action = 'store', default = [1, 10],
                        help = 'Prediction start/stop indices.')

    parser.add_argument('-l', '--lib', nargs = '+',
                        dest   = 'lib', type = int, 
                        action = 'store', default = [1, 10],
                        help = 'Library start/stop indices.')

    parser.add_argument('-mE', '--maxE',
                        dest   = 'maxE', type = int, 
                        action = 'store', default = 10,
                        help = 'Max Embedding dimension.')

    parser.add_argument('-E', '--EmbedDimension',
                        dest   = 'E', type = int, 
                        action = 'store', default = 2,
                        help = 'Embedding dimension.')

    parser.add_argument('-D', '--MultiviewDimension',
                        dest   = 'D', type = int, 
                        action = 'store', default = 2,
                        help = 'Multiview dimension.')

    parser.add_argument('-k', '--knn',
                        dest   = 'knn', type = int, 
                        action = 'store', default = 0,
                        help = 'Number of nearest neighbors.')

    parser.add_argument('-T', '--Tp',
                        dest   = 'Tp', type = int, 
                        action = 'store', default = 1,
                        help = 'Forecast interval (1 default).')

    parser.add_argument('-mT', '--maxTp',
                        dest   = 'maxTp', type = int, 
                        action = 'store', default = 10,
                        help = 'Max Predict interval.')

    parser.add_argument('-u', '--tau',
                        dest   = 'tau', type = int, 
                        action = 'store', default = -1,
                        help = 'Time delay (tau).')

    parser.add_argument('-c', '--columns', # nargs = '*',
                        dest   = 'columns', type = str,
                        action = 'store', default = '',
                        help = 'Data or embedded data column names.')

    parser.add_argument('-r', '--target',
                        dest   = 'target', type = str,
                        action = 'store', default = '',
                        help = 'Data library target column name.')

    parser.add_argument('-t', '--theta',
                        dest   = 'theta', type = float, 
                        action = 'store', default = 0,
                        help = 'S-Map local weighting exponent (0 default).')

    parser.add_argument('-ts', '--thetas',
                        dest   = 'thetas', type = str, 
                        action = 'store',
                        default = "0.01 0.05 0.1 0.2 0.4 0.6 0.8 1 1.5 " +\
                                  "2 2.5 3 3.5 4 5 6 7 8 9 10 11 12 13 14 15",
                        help = 'Predict Nonlinear thetas.')

    parser.add_argument('-x', '--exclusionRadius',
                        dest   = 'exclusionRadius', type = int, 
                        action = 'store', default = 0,
                        help = 'Time vector exclusion radius (0 default).')

    parser.add_argument('-M', '--multiview',
                        dest   = 'multiview', type = int, 
                        action = 'store', default = 0,
                        help = 'Multiview ensemble size (sqrt(m) default).')

    parser.add_argument('-tl', '--trainLib',
                        dest   = 'trainLib',
                        action = 'store_false', default = True,
                        help = 'Use in-sample (lib == pred) prediction.')

    parser.add_argument('-et', '--excludeTarget',
                        dest   = 'excludeTarget',
                        action = 'store_true', default = False,
                        help = 'Exclude target variable from multiviews.')

    parser.add_argument('-e', '--embedded',
                        dest   = 'embedded',
                        action = 'store_true', default = False,
                        help = 'Input data is an embedding.')

    parser.add_argument('-L', '--libSize', nargs = '*',
                        dest   = 'libSize', type = int,
                        action = 'store',
                        default = [ 10, 80, 10 ],
                        help = 'CCM Library size range [start, stop, incr].')

    parser.add_argument('-s', '--subsample',
                        dest   = 'subsample', type = int, 
                        action = 'store',      default = 100,
                        help = 'Number subsamples generated at each library.')

    parser.add_argument('-R', '--random',
                        dest   = 'random', 
                        action = 'store_true', default = False,
                        help = 'CCM random library samples enabled.')

    parser.add_argument('-rp', '--replacement',
                        dest   = 'replacement', 
                        action = 'store_true', default = False,
                        help = 'CCM random library with replacement: ' +\
                               '(False default).')

    parser.add_argument('-id', '--includeData',
                        dest   = 'includeData', 
                        action = 'store_true', default = False,
                        help = 'CCM: Include Data from each model: ' +\
                               '(False default).')

    parser.add_argument('-S', '--seed',
                        dest   = 'seed', type = int, 
                        action = 'store',      default = 0,
                        help = 'Random number generator seed: (None default)')

    parser.add_argument('-g', '--generateSteps',
                        dest   = 'generateSteps', type = int, 
                        action = 'store', default = 0,
                        help = 'Prediction feedback generative steps.')

    parser.add_argument('-ce', '--CE',
                        dest   = 'CE', type = str, 
                        action = 'store', default = 0,
                        help = 'CE conditional embedding expression.')

    parser.add_argument('-nt', '--nThreads',
                        dest   = 'nThreads', type = int, 
                        action = 'store', default = 4,
                        help = 'Number of threads.')

    parser.add_argument('-df', '--dataFrameName',
                        dest   = 'dataFrameName', type = str, 
                        action = 'store',      default = None,
                        help = 'Input DataFrame variable name.')

    parser.add_argument('-pa', '--path',
                        dest   = 'path', type = str, 
                        action = 'store',      default = './',
                        help = 'Input & Output file path.')

    parser.add_argument('-i', '--inputFile',
                        dest   = 'inputFile', type = str, 
                        action = 'store',     default = '../data/Lorenz5D.csv',
                        help = 'Input observation file.')

    parser.add_argument('-o', '--outputFile',
                        dest   = 'outputFile', type = str, 
                        action = 'store',      default = None,
                        help = 'Output prediction file.')
    
    parser.add_argument('-os', '--outputSmapFile',
                        dest   = 'outputSmapFile', type = str, 
                        action = 'store',      default = None,
                        help = 'S-map Output file.')
    
    parser.add_argument('-oe', '--outputEmbed',
                        dest   = 'outputEmbed', type = str, 
                        action = 'store',      default = None,
                        help = 'Output embedded data file.')
    
    parser.add_argument('-fs', '--figureSize', nargs = 2,
                        dest   = 'figureSize', type = float,
                        action = 'store', default = [ 4, 2.5 ],
                        help = 'Figure size (default [5, 3]).')
    
    parser.add_argument('-P', '--plot',
                        dest   = 'plot',
                        action = 'store_true', default = False,
                        help = 'Show plot.')
    
    parser.add_argument('-PT', '--plotTitle',
                        dest   = 'plotTitle', type = str,
                        action = 'store', default = None,
                        help = 'Plot title.')
    
    parser.add_argument('-PX', '--plotXLabel',
                        dest   = 'plotXLabel', type = str,
                        action = 'store', default = 'Time ()',
                        help = 'Plot x-axis label.')
    
    parser.add_argument('-PY', '--plotYLabel',
                        dest   = 'plotYLabel', type = str,
                        action = 'store', default = 'Amplitude ()',
                        help = 'Plot y-axis label.')
    
    parser.add_argument('-PD', '--plotDate',  # Set automatically 
                        dest   = 'plotDate',
                        action = 'store_true', default = False,
                        help = 'Time values are pyplot datetime numbers.')
    
    parser.add_argument('-v', '--verbose',
                        dest   = 'verbose',
                        action = 'store_true', default = False,
                        help = 'Print status messages.')
    
    parser.add_argument('-w', '--warnings',
                        dest   = 'warnings',
                        action = 'store_true', default = False,
                        help = 'Show warnings.')
    
    parser.add_argument('-d', '--debug',
                        dest   = 'Debug',
                        action = 'store_true', default = False,
                        help = 'Activate Debug messsages.')
    
    args = parser.parse_args()

    # Convert figureSize to a tuple
    args.figureSize = tuple( args.figureSize )

    if args.Debug:
        print( 'ParseCmdLine()' )
        print( args )

    return args
