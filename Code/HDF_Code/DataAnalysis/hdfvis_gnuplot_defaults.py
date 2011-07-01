# Defaults for gnuplot
hdfvis_gnuplot_anima=[\
          'set style data lines',\
          'set grid',\
          'set xlabel "r"',\
          'set ylabel "value of function"'\
          ]
hdfvis_gnuplot_plot = hdfvis_gnuplot_anima
hdfvis_gnuplot_error = [\
    'set xlabel "r"',\
    'set ylabel "Log base 2 of absolute value of difference"',\
    'set y2tics nomirror',\
    'set y2tics',\
    'set style data lines',\
    'set grid',\
    'set key left'\
    ]

# Defaults for gnuplot output
# Please ensure that the file extension is appropriate for the terminal
hdfvis_gnuplot_anima_nodis = ['set terminal gif animate optimize']
hdfvis_gnuplot_anima_nodis_ext = 'gif'
hdfvis_gnuplot_plot_nodis  = ['set terminal postscript eps enhanced color']
hdfvis_gnuplot_plot_nodis_ext = 'eps'
hdfvis_gnuplot_error_nodis  = ['set terminal postscript eps enhanced color']
hdfvis_gnuplot_error_nodis_ext = 'eps'
