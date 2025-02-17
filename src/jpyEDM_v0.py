
# Python distribution modules
import sys
from   io          import BytesIO
from   os          import environ
from   collections import OrderedDict
from   time        import sleep

# Community modules
from   IPython.display   import display
from   IPython.display   import clear_output as clearDashboard
import ipywidgets        as     widgets
from   matplotlib        import pyplot as plt # Explicit plt.show in Output

from   pandas            import DataFrame, read_csv, to_datetime
from   pyEDM             import *
from   pyEDM             import __version__     as pyEDMVersion
from   pyEDM             import __versionDate__ as pyEDMVersionDate

from IPython.core.getipython import get_ipython

sklearnImported = None
try:
    from sklearn.linear_model import Ridge,   Lasso,   ElasticNet
    from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
except ImportError:
    sklearnImported = False
else:
    sklearnImported = True

# Patch sys.path so local modules are found in ../
# Alternative is to set the JUPYTER_PATH environment variable 
# sys.path.append("../")
# environ["JUPYTER_PATH"] = "../"

# Local modules
from src.ArgParse        import ParseCmdLine
from src.MI_tau          import MI_tau
from src.PlotFunctions   import *

# JP: OO class implementation would allow separation across files
# from src.pyEDM_Interface import *
# from src.AuxFunctions    import *

# Monkey patch sys.argv so that parser.parse_args() doesn't choke
# on invalid arguments from the IPython/Jupyter invocation
sys.argv = ['Jupyter pyEDM']

# Globals # JP: OO would get rid of these, at least they are in EDM namespace
args    = ParseCmdLine()
Widgets = OrderedDict() # Dictionary of arg names and widgets

dataFrameIn     = None # Pandas DataFrame input
dataFrameOut    = None # Output from pyEDM
validLib        = []   # Simplex, SMap, Eval functions CE
SMapSolver      = None # SMap
targetCCMOrder  = []   # Order of target selection
columnCCMOrder  = []   # Order of column selection
plotSelectOrder = []   # Order of column selection

dashboard = widgets.VBox() # [ argumentWidgets, controlWidgets, outputTab ]

dfInput      = widgets.Output()
dfOutput     = widgets.Output()
Plot2DOutput = widgets.Output()
Plot3DOutput = widgets.Output()

outputTab = widgets.Tab( [ dfInput, dfOutput, Plot2DOutput, Plot3DOutput ] )
outputTab.set_title( 0, 'Input'   )
outputTab.set_title( 1, 'Output'  )
outputTab.set_title( 2, '2D Plot' )
outputTab.set_title( 3, '3D Plot' )

version = "Version 0.9.1 2025-02-05" + \
          "  pyEDM: " + pyEDMVersion + " " + pyEDMVersionDate +\
          "  ipywidgets: " + widgets.__version__

#============================================================================
def Version():
    with dfOutput :
        display( print( version ) )

#============================================================================
def RunButtonClicked( b ):
    '''Call the wrapper interface to pyEDM for the specified method'''
    global dataFrameOut
    global validLib

    dfOutput.clear_output()
    UpdateArgs()

    Widgets['running'].value = True # Display checkbox for busy EDM
    sleep( 0.05 ) # Let the running widget update before calls/threads

    method = Widgets['method'].value # Simplex SMap CCM...

    if len( args.CE ) and \
       method in ['Simplex', 'SMap', 'Embed Dimension',
                  'Predict Interval', 'Predict Nonlinear'] :
        validLib = dataFrameIn.eval( args.CE )
    else :
        validLib = []

    try: # Catch exceptions from pyEDM
        if 'Simplex' in method :
            dataFrameOut = Simplex_()
        elif 'SMap' in method :
            dataFrameOut = SMap_()
        elif 'CCM' in method :
            dataFrameOut = CCM_()
        elif 'Multiview' in method :
            dataFrameOut = Multiview_()
        elif 'Embed Dimension' in method :
            dataFrameOut = EmbedDimension_()
        elif 'Predict Interval' in method :
            dataFrameOut = PredictInterval_()
        elif 'Predict Nonlinear' in method :
            dataFrameOut = PredictNonlinear_()
        elif 'Embed' in method :
            dataFrameOut = Embed_()
        elif 'Mutual Information' in method :
            dataFrameOut = MutualInfo()
        elif 'Data' in method :
            ViewData()
        else :
            with dfOutput:
                display( print( "Invalid method" ) )

    except ( RuntimeError, OSError, Warning ) as error:
        dfOutput.clear_output()
        with dfOutput :
            display( print( error ) )
        Plot2DOutput.clear_output()
        with Plot2DOutput :
            display( print( error ) )
        Plot3DOutput.clear_output()
        with Plot3DOutput :
            display( print( error ) )

        Widgets['running'].value = False

    Widgets['running'].value = False

#============================================================================
def onMethodChange( change ):
    '''Refresh dashboard according to method Dropdown widget value'''

    newMethod = change['new']
    clearDashboard()
    dfOutput.clear_output()    
    UpdateArgs()

    if 'Simplex' in newMethod :
        SimplexDashboard()
    elif 'SMap' in newMethod :
        SMapDashboard()
    elif 'CCM' in newMethod :
        CCMDashboard()
    elif 'Multiview' in newMethod :
        MultiviewDashboard()
    elif 'Embed Dimension' in newMethod :
        EmbedDimensionDashboard()
    elif 'Predict Interval' in newMethod :
        PredictIntervalDashboard()
    elif 'Predict Nonlinear' in newMethod :
        PredictNonlinearDashboard()
    elif 'Embed' in newMethod :
        EmbedDashboard()
    elif 'Data' in newMethod :
        DataDashboard()
    elif 'Mutual Information' in newMethod :
        MutualInfoDashboard()
    else :
        DataDashboard()

#============================================================================
def onColumnCCMClick( change ):
    '''Save columnCCM click order into columnCCMOrder'''
    global columnCCMOrder

    if len( change['new'] ) == 1 :
        columnCCMOrder = list( change['new'] )
        return

    for column in change['new']:
        if column not in columnCCMOrder:
            columnCCMOrder.append( column )

    for column in columnCCMOrder:
        if column not in change['new']:
            columnCCMOrder.remove( column )

#============================================================================
def onTargetCCMClick( change ):
    '''Save targetCCM click order into targetCCMOrder'''
    global targetCCMOrder

    if len( change['new'] ) == 1 :
        targetCCMOrder = list( change['new'] )
        return

    for target in change['new']:
        if target not in targetCCMOrder:
            targetCCMOrder.append( target )

    for target in targetCCMOrder:
        if target not in change['new']:
            targetCCMOrder.remove( target )

#============================================================================
def onPlotSelectClick( change ):
    '''Save plotSelect click order into plotSelectOrder'''
    global plotSelectOrder

    if len( change['new'] ) == 1 :
        plotSelectOrder = list( change['new'] )
        return

    for column in change['new']:
        if column not in plotSelectOrder:
            plotSelectOrder.append( column )

    for column in plotSelectOrder:
        if column not in change['new']:
            plotSelectOrder.remove( column )

#============================================================================
def onSolverChange( change ):
    '''Call NewSolver'''
    NewSolver( change['new'] ) # Solver name in change['new']

def onAlphaChange( change ):
    '''Call NewSolver'''
    NewSolver( Widgets['solver'].value ) # alpha float in change['new']

def onL1_ratioChange( change ):
    '''Call NewSolver'''  
    NewSolver( Widgets['solver'].value ) # l1_ratio text in change['new']

#============================================================================
def NewSolver( newSolver ):
    '''Instantiate sklearn solver based on solver and parameters'''
    global SMapSolver
    
    if not sklearnImported :
        solver = None
        dfOutput.clear_output()
        with dfOutput:
            display( print( "sklearn module not imported, using SVD solver." ) )
        return

    UpdateArgs()

    alpha = Widgets['alpha'].value # FloatText widget

    # Convert l1_ratio text string to float []
    l1_ratio = [ float(x) for x in Widgets['l1_ratio'].value.split() ]
    # JP: Elastic Net Kludge since alpha is scalar.
    #     alpha and l1_ratio need to be same dimension, we have alpha scalar
    #     Need to generalize alpha to Text/vector?
    if len( l1_ratio ) == 1 :
        l1_ratio = l1_ratio[0]

    if 'SVD' in newSolver :
        SMapSolver = None
    elif 'Ridge CV' in newSolver :
        SMapSolver = RidgeCV()
    elif 'Lasso CV' in newSolver :
        SMapSolver = LassoCV( cv = 5 )
    elif 'Elastic Net CV' in newSolver :
        SMapSolver = ElasticNetCV( l1_ratio = l1_ratio, cv = 5 )
    elif 'Ridge' in newSolver :
        SMapSolver = Ridge( alpha = alpha )
    elif 'Lasso' in newSolver :
        SMapSolver = Lasso( alpha = alpha )
    elif 'Elastic Net' in newSolver :
        SMapSolver = ElasticNet( alpha = alpha, l1_ratio = l1_ratio )
    else :
        SMapSolver = None

#============================================================================
def DataPlotButtonClicked( b = None ):
    '''Explicitly call Data or Embed Plots'''

    plt.close('all') # Jupyter calls plt.show, but not close
    UpdateArgs()

    columnList = Widgets['plotSelect'].value

    if Widgets['plot_2D_3D'].value == '2D' :
        Plot2DOutput.clear_output()
        if args.scatter : kind = 'scatter'
        else :            kind = 'line'
        with Plot2DOutput :
            dataFrameIn.plot(plotSelectOrder[0], plotSelectOrder[1:], kind=kind)
            plt.show()
        display( Plot2DOutput )

    elif Widgets['plot_2D_3D'].value == '3D' :
        Plot3DOutput.clear_output()
        with Plot3DOutput :
            Plot3D( dataFrameIn, columnList, args )
            plt.show()
        display( Plot3DOutput )

#============================================================================
def onFileUploadChange( b = None ):
    '''Import dataFrameIn with FileUpload widget

       ==================================================================
       JP Note !!! : As of 2023-05-10 :
       FileUpload widget fails on large files (24k rows x 300 columns)
         https://github.com/jupyter-widgets/ipywidgets/issues/2522
         Tornado, websocket messages have a limit of 10MB
       cosole error : Restoring connection... but no return value/error
       Use ImportButtonClicked() instead with fileImport name & read_csv
       ==================================================================

       The FileUpload widget changed significantly in ipywidgets 8:
          The .value traitlet is now a list (tuple) of dictionaries,
          rather than a dictionary mapping uploaded name to content.

          The default is multiple = False :
          FileUpload( accept = ' ',  multiple = False )
          So only one file will be returned by default in tuple[0].'''

    global dataFrameIn

    # Crazy unpacking of FileUpload widget return...
    fileUploadObj   = Widgets['fileUpload']
    fileUploadValue = fileUploadObj.value # <class 'tuple'>

    if isinstance( fileUploadValue, tuple ):
        # Take [0] since FileUpload( multiple = False )
        content = fileUploadValue[0][ 'content' ]

    elif isinstance( fileUploadValue, dict ):
        # If fileUploadValue is dictionary from ipywidgets < 8:
        fileNamekey = list( fileUploadValue.keys() )[0]
        content     = fileUploadValue[ fileNamekey ][ 'content' ]

    else :
        with dfOutput :
            display(print('fileUpload widget failed. Perhaps file too large. '+\
                          'Try manual file import with explicit file path'))

    dataFrameIn = read_csv( BytesIO( content ) )

    with dfOutput :
        print( 'fileUpload dataFrameIn ' +\
               f'{dataFrameIn.shape} {dataFrameIn.columns[:3]}' )

    # Set file name in fileImport text box
    Widgets['fileImport'].value = fileUploadObj.value[0][ 'name' ]

    UpdateArgs()
    RefreshData()

    # Crazy hack to reset the FileUpload widget counter
    Widgets['fileUpload']._counter = 0

#============================================================================
def ImportButtonClicked( b ):
    '''Upload DataFrame manually since FileUpload widget fails on large files'''

    global dataFrameIn

    # JP Workaround for FileUpload widget fails on large files
    fileName    = Widgets['fileImport'].value
    dataFrameIn = read_csv( fileName )

    RefreshData()

#============================================================================
def RefreshData():
    '''Update parameters for onFileUploadChange(), or,
       Update parameters for new data assigned externally:
       EDM.dataFrameIn = EDM.read_csv('../data/TTTF.csv')'''

    dfInput.clear_output()
    with dfInput :
        display( dataFrameIn )

    # Refresh widget selectors for new dataFrameIn
    if not dataFrameIn.empty:
        Widgets[ 'plotSelect' ].options = dataFrameIn.columns
        Widgets[ 'pred'       ].min     = 1
        Widgets[ 'pred'       ].max     = dataFrameIn.shape[0]
        Widgets[ 'lib'        ].min     = 1
        Widgets[ 'lib'        ].max     = dataFrameIn.shape[0]
        Widgets[ 'columns'    ].options = dataFrameIn.columns[1:]
        Widgets[ 'target'     ].options = dataFrameIn.columns[1:]
        Widgets[ 'columnCCM'  ].options = dataFrameIn.columns[1:]
        Widgets[ 'targetCCM'  ].options = dataFrameIn.columns[1:]

        UpdateArgs()

#============================================================================
def Dashboard():
    '''Create notebook widgets to hold args parameters'''

    runButton = widgets.Button( description = "Run" )
    runButton.on_click( RunButtonClicked )

    importButton = widgets.Button( description = "Import" )
    importButton.on_click( ImportButtonClicked )

    dataPlotButton = widgets.Button( description = "Plot" )
    dataPlotButton.on_click( DataPlotButtonClicked )

    method = widgets.Dropdown( options=[ 'Simplex',
                                         'SMap',
                                         'CCM',
                                         'Multiview',
                                         'Embed Dimension',
                                         'Predict Interval',
                                         'Predict Nonlinear',
                                         'Embed',
                                         'Mutual Information',
                                         'Data' ],
                               value='Data', description='method' )
    # Callback function on method Dropdown to change dashboard
    method.observe( onMethodChange, names = 'value' )

    solver = widgets.Dropdown( options=[ 'SVD',
                                         'Ridge',
                                         'Lasso',
                                         'Elastic Net',
                                         'Ridge CV',
                                         'Lasso CV',
                                         'Elastic Net CV' ],
                               value='SVD', description='solver' )

    # Callback function on solver Dropdown to instantiate solver
    solver.observe( onSolverChange, names = 'value' )

    lib = widgets.IntRangeSlider( value=[1,500], min=1, max=1000, step=1,
                                  description='lib' )

    pred = widgets.IntRangeSlider( value=[501,600], min=1, max=1000, step=1,
                                   description='pred' )

    columns = widgets.SelectMultiple( description='columns', rows = 3 )

    target = widgets.Dropdown( options=[], description='target' )

    # CCM : preserve click order to help in labeling : 1st click is label
    columnCCM = widgets.SelectMultiple( description='columns', rows = 3 )
    targetCCM = widgets.SelectMultiple( description='target',  rows = 3 )
    # Callback function on targetCCM SelectMultiple to save click order
    columnCCM.observe( onColumnCCMClick, names = 'value' )
    targetCCM.observe( onTargetCCMClick, names = 'value' )

    tau = widgets.IntSlider( value=-1, min=-50, max=50, step=1,
                             description='tau')

    knn = widgets.IntSlider( value=0, min=0, max=50, step=1,
                             description='knn' )

    E = widgets.IntSlider( value=3, min=0, max=50, step=1, description='E' )

    maxE = widgets.IntSlider( value=10, min=5, max=50, step=1,
                              description='maxE' )

    Tp = widgets.IntSlider( value=1, min=-50, max=50, step=1, description='Tp' )

    maxTp = widgets.IntSlider( value=10, min=5, max=50, step=1,
                               description='maxTp' )

    exclusionRadius = widgets.IntSlider( value=0, min=0, max=100, step=1,
                                         description='exclRad')

    theta = widgets.FloatText( value=3, description='theta',
                               layout = widgets.Layout(width='50%') )

    thetas = widgets.Text( value=args.thetas, description='thetas',
                           layout = widgets.Layout(width='100%') )

    alpha = widgets.FloatText( value=0.05, description='alpha',
                               layout = widgets.Layout(width='50%') )
    # Callback function on alpha to instantiate solver
    alpha.observe( onAlphaChange, names = 'value' )

    l1_ratio = widgets.Text( value='0.05', description='l1_ratio',
                             layout = widgets.Layout(width='90%') )
    # Callback function on l1_ratio to instantiate solver
    l1_ratio.observe( onL1_ratioChange, names = 'value' )

    generateSteps = widgets.IntText( value=0, description='generate',
                                     layout = widgets.Layout(width='50%') )

    CE = widgets.Text( value='', description='CE' )

    D = widgets.IntSlider( value=0, min=0, max=15, step=1, description='D' )

    multiview = widgets.IntSlider( value=0, min=0, max=20, step=1,
                                   description='multiview' )

    libsize = widgets.Text( value='20 200 20', description='libsize')

    sample = widgets.IntSlider( value=100, min=5, max=200, step=1,
                                description='sample' )

    seed = widgets.IntText( value=None, description='seed' )

    maxLag = widgets.IntSlider( value=15, min=0, max=50, step=1,
                                description='maxLag' )

    MI_neighbors = widgets.IntSlider( value=10, min=2, max=100, step=1,
                                      description='MI_neighbors' )

    nProcess = widgets.IntText( value=4, description='nProcess' )

    # outputFile = widgets.Text( value='', description='Prediction file')
    # outputSmapFile = widgets.Text( value='', description='S-Map Output',
    #                                style = {'description_width' : 'initial'} )
    # outputEmbed = widgets.Text( value='', description='Embed Output',
    #                             indent = False, 
    #                             style = {'description_width' : 'initial'} )

    # Label in dashboard
    plotSelect = widgets.SelectMultiple( description = '' )
    # Callback function on plotSelect SelectMultiple to save click order
    plotSelect.observe( onPlotSelectClick, names = 'value' )

    plot_2D_3D = widgets.Dropdown( options = ['2D','3D'], value = '2D',
                                   description = '2D or 3D',
                                   layout = widgets.Layout(width='45%'))

    # JP FileUpload() widget does not work on large files (>10 MB)
    fileUpload = widgets.FileUpload( accept = '.csv', multiple = False )
    # Callback function on fileUpload
    fileUpload.observe( onFileUploadChange, names = 'value' )

    # JP Instead, read file name from fileImport TextBox
    # Callback function is on ImportButtonClicked to load fileUpload.value
    fileImport = widgets.Text( value='', description='', # Label in dashboard
                               style = {'description_width' : 'initial'} )

    # Checkbox ----------------------------------------------------------
    plot = widgets.Checkbox( value = True, description='plot',
                             indent = False,
                             layout = widgets.Layout(width='80%') )

    scatter = widgets.Checkbox( value = False, description='scatter plot',
                                indent = False,
                                layout = widgets.Layout(width='60%') )

    trainLib = widgets.Checkbox( value = True, description='trainLib',
                                 indent = False,
                                 layout = widgets.Layout(width='60%') )

    embedded = widgets.Checkbox( value = False, description='embedded',
                                 indent = False,
                                 layout = widgets.Layout(width='100%') )

    noTime = widgets.Checkbox( value = False, description='noTime',
                               indent = False,
                               layout = widgets.Layout(width='100%') )

    #randomLib = widgets.Checkbox( value = True, description='randomLib',
    #                              indent = False,
    #                              layout = widgets.Layout(width='60%') )
    #replacement = widgets.Checkbox( value = False, description='replacement',
    #                                indent = False,
    #                                layout = widgets.Layout(width='60%') )

    excludeTarget = widgets.Checkbox( value = False, description='excludeTarget',
                                      indent = False,
                                      layout = widgets.Layout(width='60%') )

    generateConcat = widgets.Checkbox( value = False,
                                       description = 'generateConcat',
                                       indent = False,
                                       layout = widgets.Layout(width='100%') )

    running = widgets.Checkbox( value = False, description='Running',
                                indent = False,
                                layout = widgets.Layout(width='60%') )

    verbose = widgets.Checkbox( value = False, description='verbose',
                                indent = False,
                                layout = widgets.Layout(width='100%') )

    # Populate global dictionary of widgets
    Widgets['runButton']       = runButton
    Widgets['fileUpload']      = fileUpload
    Widgets['fileImport']      = fileImport
    Widgets['importButton']    = importButton
    Widgets['dataPlotButton']  = dataPlotButton
    Widgets['method']          = method
    Widgets['solver']          = solver
    Widgets['lib']             = lib
    Widgets['pred']            = pred
    Widgets['E']               = E
    Widgets['maxE']            = maxE
    Widgets['D']               = D
    Widgets['knn']             = knn
    Widgets['Tp']              = Tp
    Widgets['maxTp']           = maxTp
    Widgets['exclusionRadius'] = exclusionRadius
    Widgets['theta']           = theta
    Widgets['thetas']          = thetas
    Widgets['alpha']           = alpha
    Widgets['l1_ratio']        = l1_ratio
    Widgets['tau']             = tau
    Widgets['columns']         = columns
    Widgets['target']          = target
    Widgets['columnCCM']       = columnCCM
    Widgets['targetCCM']       = targetCCM
    Widgets['embedded']        = embedded
    Widgets['generateSteps']   = generateSteps
    Widgets['generateConcat']  = generateConcat
    Widgets['CE']              = CE
    Widgets['multiview']       = multiview
    Widgets['trainLib']        = trainLib
    Widgets['excludeTarget']   = excludeTarget
    Widgets['libsize']         = libsize
    Widgets['sample']          = sample
    Widgets['noTime']          = noTime
    #Widgets['randomLib']      = randomLib
    #Widgets['replacement']    = replacement
    Widgets['seed']            = seed
    Widgets['maxLag']          = maxLag
    Widgets['MI_neighbors']    = MI_neighbors
    Widgets['nProcess']        = nProcess
    #Widgets['outputFile']     = outputFile
    #Widgets['outputSmapFile'] = outputSmapFile
    #Widgets['outputEmbed']    = outputEmbed
    Widgets['plotSelect']      = plotSelect
    Widgets['plot_2D_3D']      = plot_2D_3D
    Widgets['plot']            = plot
    Widgets['scatter']         = scatter
    Widgets['verbose']         = verbose
    Widgets['running']         = running

    # Load initial data
    global dataFrameIn
    dataFrameIn = ReadCSV( args.inputFile ) 
    RefreshData()

    DataDashboard() # Default start up
    Version()

#============================================================================
def UpdateArgs():
    '''Update EDM args parameters from notebook widgets.
       NOTE that some widget parameters are not stored in args.
       For example: CE, alpha, l1_ratio'''
    global args

    args.method          = Widgets['method'].value
    args.solver          = Widgets['solver'].value
    args.pred            = Widgets['pred'].value
    args.lib             = Widgets['lib'].value
    args.E               = Widgets['E'].value
    args.maxE            = Widgets['maxE'].value
    args.D               = Widgets['D'].value
    args.knn             = Widgets['knn'].value
    args.Tp              = Widgets['Tp'].value
    args.maxTp           = Widgets['maxTp'].value
    args.exclusionRadius = Widgets['exclusionRadius'].value
    args.theta           = Widgets['theta'].value
    args.thetas          = Widgets['thetas'].value
    args.tau             = Widgets['tau'].value
    args.columns         = list( Widgets['columns'].value )
    args.target          = Widgets['target'].value
    args.columnCCM       = list( Widgets['columnCCM'].value )
    args.targetCCM       = list( Widgets['targetCCM'].value )
    args.multiview       = Widgets['multiview'].value
    args.trainLib        = Widgets['trainLib'].value
    args.excludeTarget   = Widgets['excludeTarget'].value
    args.embedded        = Widgets['embedded'].value
    args.generateSteps   = Widgets['generateSteps'].value
    if args.generateSteps < 0 :
        args.generateSteps = Widgets['generateSteps'].value = 0

    args.generateConcat  = Widgets['generateConcat'].value
    args.CE              = Widgets['CE'].value
    #args.libSize        = [ int(x) for x in Widgets['libsize'].value.split() ]
    args.libSize         = Widgets['libsize'].value
    args.sample          = Widgets['sample'].value
    args.noTime          = Widgets['noTime'].value
    #args.random         = Widgets['randomLib'].value
    #args.replacement    = Widgets['replacement'].value
    args.seed            = Widgets['seed'].value
    args.maxLag          = Widgets['maxLag'].value
    args.MI_neighbors    = Widgets['MI_neighbors'].value

    args.nProcess        = Widgets['nProcess'].value
    if args.nProcess < 1 :
        args.nProcess = Widgets['nProcess'].value = 1

    args.inputFile       = '' # fileUpload / Import directly assigns dataFrameIn
    #args.outputFile     = Widgets['outputFile'].value
    #args.outputSmapFile = Widgets['outputSmapFile'].value
    #args.outputEmbed    = Widgets['outputEmbed'].value
    args.plot            = Widgets['plot'].value
    args.scatter         = Widgets['scatter'].value
    args.verbose         = Widgets['verbose'].value
    args.running         = Widgets['running'].value

#============================================================================
def RenderDashboard( left_box, mid_box, right_box ):
    '''Display dashboard with consistent layout'''

    controlWidgets  = widgets.HBox([ Widgets['runButton'], Widgets['running'] ])
    argumentWidgets = widgets.HBox( [ left_box, mid_box, right_box ] )
    dashboard.children = [ argumentWidgets, controlWidgets, outputTab ]

    display( dashboard )

#============================================================================
def DataDashboard():
    '''Select Data widgets'''

    Widgets[ 'plotSelect' ].options = dataFrameIn.columns

    # Organize widgets
    left_box  = widgets.VBox( [ Widgets['method'],
                                Widgets['scatter'],
                                Widgets['plot_2D_3D'],
                                Widgets['dataPlotButton'] ] )
    
    mid_box   = widgets.VBox( [ widgets.Label( 'Plot Columns:' ),
                                Widgets['plotSelect'] ] )
    
    right_box = widgets.VBox( [ widgets.Label( 'Upload files < 10MB:' ),
                                Widgets['fileUpload'],
                                widgets.Label( 'Manual Import files > 10MB:' ),
                                widgets.HBox( [ Widgets['fileImport'] ] ),
                                Widgets['importButton'] ] )

    RenderDashboard( left_box, mid_box, right_box )

#============================================================================
def MutualInfoDashboard():
    '''Select Mutual Info widgets'''

    # Organize widgets
    left_box  = widgets.VBox( [ Widgets['method'],
                                Widgets['columns'], Widgets['target'] ] )

    mid_box   = widgets.VBox( [ Widgets['maxLag'],
                                Widgets['tau'], Widgets['MI_neighbors'] ] )

    right_box = widgets.VBox( [ Widgets['plot'], Widgets['verbose'] ] )

    RenderDashboard( left_box, mid_box, right_box )

#============================================================================
def EmbedDashboard():
    '''Select Embed Dimension widgets'''

    # Organize widgets 
    left_box  = widgets.VBox( [ Widgets['method'], Widgets['columns'] ] )

    mid_box   = widgets.VBox( [ Widgets['E'], Widgets['tau'],
                                Widgets['plotSelect'] ] )

    right_box = widgets.VBox( [ # Widgets['outputEmbed'],
                                Widgets['plot'], Widgets['verbose'] ] )

    RenderDashboard( left_box, mid_box, right_box )

#============================================================================
def PredictNonlinearDashboard():
    '''Select Predict Nonlinear widgets'''

    # Organize widgets 
    left_box  = widgets.VBox( [ Widgets['method'],
                                Widgets['lib'],     Widgets['pred'],
                                Widgets['columns'], Widgets['target'] ] )

    mid_box   = widgets.VBox( [ Widgets['E'],   Widgets['Tp'], 
                                Widgets['tau'], Widgets['thetas'],
                                Widgets['exclusionRadius'], Widgets['CE'] ] )

    right_box = widgets.VBox( [ #Widgets['outputFile'],
                                Widgets['nProcess'], Widgets['plot'],
                                Widgets['embedded'], Widgets['noTime'],
                                Widgets['verbose'] ] )

    RenderDashboard( left_box, mid_box, right_box )

#============================================================================
def PredictIntervalDashboard():
    '''Select Predict Interval widgets'''

    # Organize widgets 
    left_box  = widgets.VBox( [ Widgets['method'],
                                Widgets['lib'],     Widgets['pred'],
                                Widgets['columns'], Widgets['target'] ] )

    mid_box   = widgets.VBox( [ Widgets['maxTp'],
                                Widgets['E'], Widgets['tau'],
                                Widgets['exclusionRadius'], Widgets['CE'] ] )

    right_box = widgets.VBox( [ #Widgets['outputFile'],
                                Widgets['nProcess'], Widgets['plot'],
                                Widgets['embedded'], Widgets['noTime'],
                                Widgets['verbose'] ] )

    RenderDashboard( left_box, mid_box, right_box )

#============================================================================
def EmbedDimensionDashboard():
    '''Select Embed Dimension widgets'''

    # Organize widgets 
    left_box  = widgets.VBox( [ Widgets['method'],
                                Widgets['lib'],     Widgets['pred'],
                                Widgets['columns'], Widgets['target'] ] )

    mid_box   = widgets.VBox( [ Widgets['maxE'],
                                Widgets['Tp'],  Widgets['tau'],
                                Widgets['exclusionRadius'], Widgets['CE'] ] )

    right_box = widgets.VBox( [ #Widgets['outputFile'],
                                Widgets['nProcess'], Widgets['plot'],
                                Widgets['noTime'],   Widgets['verbose'] ] )

    RenderDashboard( left_box, mid_box, right_box )

#============================================================================
def MultiviewDashboard():
    '''Select Multiview widgets'''

    # Organize widgets 
    left_box  = widgets.VBox( [ Widgets['method'],
                                Widgets['lib'],     Widgets['pred'],
                                Widgets['columns'], Widgets['target'] ] )

    mid_box   = widgets.VBox( [ Widgets['E'],  Widgets['knn'],
                                Widgets['Tp'], Widgets['tau'],
                                Widgets['exclusionRadius'],
                                Widgets['D'],  Widgets['multiview'] ] )

    right_box = widgets.VBox( [ #Widgets['outputFile'],
                                Widgets['nProcess'],   Widgets['plot'],
                                Widgets['trainLib'],   Widgets['excludeTarget'],
                                Widgets['verbose'] ] )

    RenderDashboard( left_box, mid_box, right_box )

#============================================================================
def CCMDashboard():
    '''Select CCM widgets'''

    # Organize widgets 
    left_box  = widgets.VBox( [ Widgets['method'],
                                Widgets['columnCCM'], Widgets['targetCCM'],
                                Widgets['libsize'], Widgets['seed'] ] )

    mid_box   = widgets.VBox( [ Widgets['E'],   Widgets['knn'], Widgets['Tp'],
                                Widgets['tau'], Widgets['exclusionRadius'],
                                Widgets['sample'] ] )

    right_box = widgets.VBox( [ # Widgets['outputFile'],
                                Widgets['plot'],   Widgets['embedded'],
                                # Widgets['replacement'], Widgets['randomLib'],
                                Widgets['noTime'], Widgets['verbose'] ] )

    RenderDashboard( left_box, mid_box, right_box )

#============================================================================
def SMapDashboard():
    '''Select SMap widgets'''

    # Organize widgets 
    left_box  = widgets.VBox( [ Widgets['method'],
                                Widgets['lib'],     Widgets['pred'],
                                Widgets['columns'], Widgets['target'],
                                Widgets['solver'],  Widgets['alpha'],
                                Widgets['l1_ratio'] ] )

    mid_box   = widgets.VBox( [ Widgets['theta'], Widgets['E'], Widgets['Tp'],
                                Widgets['tau'],   Widgets['knn'],
                                Widgets['exclusionRadius'],
                                Widgets['CE'],    Widgets['generateSteps']
                               ] )

    right_box = widgets.VBox( [ # Widgets['outputFile'],
                                Widgets['plot'],   Widgets['embedded'],
                                Widgets['noTime'], Widgets['generateConcat'],
                                Widgets['verbose'] ] )

    RenderDashboard( left_box, mid_box, right_box )

#============================================================================
def SimplexDashboard():
    '''Select Simplex widgets'''

    # Organize widgets 
    left_box  = widgets.VBox( [ Widgets['method'],
                                Widgets['lib'],     Widgets['pred'],
                                Widgets['columns'], Widgets['target'] ] )

    mid_box   = widgets.VBox( [ Widgets['E'],   Widgets['Tp'],
                                Widgets['tau'], Widgets['knn'],
                                Widgets['exclusionRadius'],
                                Widgets['CE'],  Widgets['generateSteps']
                               ] )

    right_box = widgets.VBox( [ #Widgets['outputFile'],
                                Widgets['plot'],   Widgets['embedded'],
                                Widgets['noTime'], Widgets['generateConcat'],
                                Widgets['verbose'] ] )

    RenderDashboard( left_box, mid_box, right_box )

#============================================================================
def ViewData():
    '''Explore data'''

    dfInput.clear_output()
    with dfInput :
        display( dataFrameIn )

    if args.plot :
        DataPlotButtonClicked()

#============================================================================
# @dfOutput.capture() # Decorator : Not used : Explicit Output context used
def MutualInfo():
    '''Mutual Information on lagged columns : target'''
    D = MI_tau( args, dataFrameIn )

    with dfOutput :
        display( D )

    if args.plot :
        plt.close('all')
        Plot2DOutput.clear_output()
        with Plot2DOutput :
            PlotMutualInfo( D, args )
            plt.show()
        display( Plot2DOutput )

    return D

#============================================================================
# @dfOutput.capture() # Decorator : Not used : Explicit Output context used
def Embed_():
    '''Interface for Embed()'''

    D = Embed( dataFrame   = dataFrameIn,
               E           = args.E,
               tau         = args.tau,
               columns     = args.columns,
               includeTime = False,
               pathIn      = '',
               dataFile    = None )

    Widgets[ 'plotSelect' ].options = D.columns
    columnList = Widgets['plotSelect'].value

    with dfOutput :
        display( D )

    if args.plot :
        plt.close('all') # Jupyter calls plt.show, but not close

        columnList = Widgets['plotSelect'].value

        if len( columnList ) == 2 :
            Plot2DOutput.clear_output()
            with Plot2DOutput :
                D.plot( columnList[0], columnList[1] )
                plt.show()
            display( Plot2DOutput )

        elif len( columnList ) == 3 :
            Plot3DOutput.clear_output()
            with Plot3DOutput :
                Plot3D( D, columnList, args )
                plt.show()
            display( Plot3DOutput )

    return D

#============================================================================
#@dfOutput.capture() # Decorator: Not used : Explicit Output context used
def EmbedDimension_():
    '''Interface for EmbedDimension()'''

    D = EmbedDimension( dataFrame   = dataFrameIn,
                        columns     = args.columns,
                        target      = args.target,
                        maxE        = args.maxE,
                        lib         = args.lib,
                        pred        = args.pred,
                        Tp          = args.Tp,
                        tau         = args.tau,
                        exclusionRadius = args.exclusionRadius,
                        embedded    = args.embedded,
                        validLib    = validLib,
                        noTime      = args.noTime,
                        ignoreNan   = True,
                        verbose     = args.verbose,
                        numProcess  = args.nProcess,
                        showPlot    = False )

    with dfOutput :
        display( D )

    if args.plot :
        plt.close('all')
        Plot2DOutput.clear_output()
        with Plot2DOutput :
            PlotEmbedDimension( D, args )
            plt.show()
        display( Plot2DOutput )

    return D

#============================================================================
#@dfOutput.capture() # Decorator: Not used : Explicit Output context used
def PredictInterval_():
    '''Interface for PredictInterval()'''

    D = PredictInterval( dataFrame   = dataFrameIn,
                         columns     = args.columns,
                         target      = args.target,
                         lib         = args.lib,
                         pred        = args.pred,
                         maxTp       = args.maxTp,
                         E           = args.E,
                         tau         = args.tau,
                         exclusionRadius = args.exclusionRadius,
                         embedded    = args.embedded,
                         validLib    = validLib,
                         noTime      = args.noTime,
                         ignoreNan   = True,
                         verbose     = args.verbose,
                         numProcess  = args.nProcess,
                         showPlot    = False )

    with dfOutput :
        display( D )

    if args.plot :
        plt.close('all')
        Plot2DOutput.clear_output()
        with Plot2DOutput :
            PlotPredictInterval( D, args )
            plt.show()
        display( Plot2DOutput )

    return D

#============================================================================
#@dfOutput.capture() # Decorator: Not used : Explicit Output context used
def PredictNonlinear_():
    '''Interface for PredictNonlinear()'''

    D = PredictNonlinear( dataFrame   = dataFrameIn,
                          columns     = args.columns,
                          target      = args.target,
                          theta       = args.thetas,
                          lib         = args.lib,
                          pred        = args.pred,
                          E           = args.E,
                          Tp          = args.Tp,
                          knn         = args.knn,
                          tau         = args.tau,
                          exclusionRadius = args.exclusionRadius,
                          solver      = None,
                          embedded    = args.embedded,
                          validLib    = validLib,
                          noTime      = args.noTime,
                          ignoreNan   = True,
                          verbose     = args.verbose,
                          numProcess  = args.nProcess,
                          showPlot    = False )

    with dfOutput :
        display( D )

    if args.plot :
        plt.close('all')
        Plot2DOutput.clear_output()
        with Plot2DOutput :
            PlotPredictNonlinear( D, args )
            plt.show()
        display( Plot2DOutput )

    return D

#============================================================================
#@dfOutput.capture() # Decorator: Not used : Explicit Output context used
def Multiview_():
    '''Interface for Multiview()'''

    if args.target not in args.columns :
        dfOutput.clear_output()
        with dfOutput :
            display( print( "Multiview: Include the target in columns" ) )
            Widgets['running'].value = False
            return None

    D = Multiview( dataFrame       = dataFrameIn,
                   columns         = args.columns,
                   target          = args.target,
                   lib             = args.lib,
                   pred            = args.pred,
                   D               = args.D,
                   E               = args.E,
                   Tp              = args.Tp,
                   knn             = args.knn,
                   tau             = args.tau,
                   multiview       = args.multiview,
                   exclusionRadius = args.exclusionRadius,
                   trainLib        = args.trainLib,
                   excludeTarget   = args.excludeTarget,
                   ignoreNan       = True,
                   verbose         = args.verbose,
                   numProcess      = args.nProcess )

    with dfOutput :
        display( D[ 'View' ] )

    if args.plot :
        plt.close('all')
        Plot2DOutput.clear_output()
        with Plot2DOutput :
            PlotObsPred_( D[ 'Predictions' ], args )
            plt.show()
        display( Plot2DOutput )

    return D

#============================================================================
#@dfOutput.capture() # Decorator: Not used : Explicit Output context used
def CCM_():
    '''Interface for CCM()'''

    D = CCM( dataFrame       = dataFrameIn,
             columns         = columnCCMOrder, # Click order, not args
             target          = targetCCMOrder, # Click order, not args
             libSizes        = args.libSize,
             sample          = args.sample,
             E               = args.E,
             Tp              = args.Tp,
             knn             = args.knn,
             tau             = args.tau,
             exclusionRadius = args.exclusionRadius,
             seed            = args.seed,
             embedded        = args.embedded,
             includeData     = args.includeData,
             noTime          = False,
             ignoreNan       = True,
             verbose         = args.verbose )

    with dfOutput :
        display( D )

    if args.plot :
        plt.close('all')
        Plot2DOutput.clear_output()
        with Plot2DOutput :
            PlotCCM( D, args )
            plt.show()
        display( Plot2DOutput )

    return D

#============================================================================
#@dfOutput.capture() # Decorator: Not used : Explicit Output context used
def SMap_():
    '''Interface for SMap()'''

    D = SMap( dataFrame       = dataFrameIn,
              columns         = args.columns,
              target          = args.target,
              lib             = args.lib,
              pred            = args.pred,
              E               = args.E,
              Tp              = args.Tp,
              knn             = args.knn,
              tau             = args.tau,
              theta           = args.theta,
              exclusionRadius = args.exclusionRadius,
              solver          = SMapSolver,
              embedded        = args.embedded,
              validLib        = validLib,
              noTime          = args.noTime,
              generateSteps   = args.generateSteps,
              generateConcat  = args.generateConcat,
              ignoreNan       = True,
              verbose         = args.verbose )

    with dfOutput :
        display( D['predictions' ] )

    if args.plot :
        plt.close('all')
        Plot2DOutput.clear_output()
        with Plot2DOutput :
            PlotSMap( D, args ) # PlotSMap() calls plt.show()
        display( Plot2DOutput )

    return D

#============================================================================
#@dfOutput.capture() # Decorator: Not used : Explicit Output context used
def Simplex_():
    '''Interface for Simplex()'''

    D = Simplex( dataFrame       = dataFrameIn,
                 columns         = args.columns,
                 target          = args.target,
                 lib             = args.lib,
                 pred            = args.pred,
                 E               = args.E,
                 Tp              = args.Tp,
                 knn             = args.knn,
                 tau             = args.tau,
                 exclusionRadius = args.exclusionRadius,
                 embedded        = args.embedded,
                 validLib        = validLib,
                 noTime          = args.noTime,
                 generateSteps   = args.generateSteps,
                 generateConcat  = args.generateConcat,
                 verbose         = args.verbose )

    with dfOutput :
        display( D )

    if args.plot :
        plt.close('all')
        Plot2DOutput.clear_output()
        with Plot2DOutput :
            PlotObsPred_( D, args )
            plt.show()
        display( Plot2DOutput )

    return D

#============================================================================
def ReadCSV( filepath = None ):
    '''Interface for read_csv() in pandas'''
    df = DataFrame()

    try:
        if filepath == None :
            UpdateArgs()
            df = read_csv( args.inputFile ) #( args.path + args.inputFile )
        else :
            df = read_csv( filepath )

        #if type( df.iloc[0,0] ) is str :
            # pyEDM handles DateTime conversion of first column
            # jpyEDM Always passes a DataFrame, not the path/file to pyEDM
            # Try to handle DateTime in first column...
            # df.iloc[:,0] = df.iloc[:,0].apply(lambda x: to_datetime(x).value)

    except ( OSError, RuntimeError, ValueError ) as error:
        dfInput.clear_output()
        with dfInput :
            display( print( error ) )
        dfOutput.clear_output()
        with dfOutput :
            display( print( error ) )

    return df

#============================================================================
def Notebook():
    '''Determine if Jupyter notebook is the parent process'''
    try:
        ipython = get_ipython().__class__.__name__
        if 'ZMQInteractiveShell' in ipython:
            return True   # Jupyter notebook or qtconsole
        elif 'TerminalInteractiveShell' in ipython:
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

#============================================================================
def IPythonVersion():
    import IPython
    return IPython.__version__
