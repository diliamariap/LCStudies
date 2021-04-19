import matplotlib.font_manager
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import roc_curve, auc
import matplotlib.gridspec as gridspec

import atlas_mpl_style as ampl
ampl.use_atlas_style()

# set plotsytle choices here
params = {'legend.fontsize': 13,
          'axes.labelsize': 18}
plt.rcParams.update(params)

ampl.set_color_cycle('Oceanic',10)

def histogramOverlay(frames, data, labels, xlabel, ylabel, figfile = '', 
                        x_min = 0, x_max = 2200, xbins = 22, normed = True, y_log = False,
                        atlas_x = -1, atlas_y = -1, simulation = False,
                        textlist = []):
    xbin = np.arange(x_min, x_max, (x_max - x_min) / xbins)

    plt.cla()
    plt.clf()
    fig = plt.figure()
    fig.patch.set_facecolor('white')

    zorder_start = -1 * len(data) # hack to get axes on top
    for i, datum in enumerate(data):
        plt.hist(frames[i][datum], bins = xbin, density = normed, 
            alpha = 0.5, label=labels[i], zorder=zorder_start + i)
    
    plt.xlim(x_min, x_max)
    if y_log:
        plt.yscale('log')

    ampl.set_xlabel(xlabel)
    ampl.set_ylabel(ylabel)

    if atlas_x >= 0 and atlas_y >= 0:
        ampl.draw_atlas_label(atlas_x, atlas_y, simulation = simulation, fontsize = 18)

    drawLabels(fig, atlas_x, atlas_y, simulation, textlist)
    
    fig.axes[0].zorder = len(data)+1 #hack to keep the tick marks up
    plt.legend()
    if figfile != '':
        plt.savefig(figfile)
    plt.show()

def lineOverlay(xcenter, lines, labels, xlabel, ylabel, figfile = '',
                    x_min = 0.1, x_max = 1000, x_log = True, y_min = 0, y_max = 2, y_log = False,
                    linestyles=[], colorgrouping=-1,
                    extra_lines = [],
                    atlas_x=-1, atlas_y=-1, simulation=False,
                    textlist=[]):
    plt.cla()
    plt.clf()

    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for extra_line in extra_lines:
        plt.plot(extra_line[0], extra_line[1], linestyle='--', color='black')

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, line in enumerate(lines):
        if len(linestyles) > 0:
            linestyle = linestyles[i]
        else:
            linestyle = 'solid'
        if colorgrouping > 0:
            color = colors[int(np.floor(i / colorgrouping))]
        else:
            color = colors[i]
        plt.plot(xcenter, line, label = labels[i], linestyle=linestyle,color=color)

    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    ampl.set_xlabel(xlabel)
    ampl.set_ylabel(ylabel)

    drawLabels(fig, atlas_x, atlas_y, simulation, textlist)

    plt.legend()
    if figfile != '':
        plt.savefig(figfile)
    plt.show()

def roc_plot(xlist, ylist, figfile = '',
             xlabel='False positive rate',
             ylabel='True positive rate',
             x_min = 0, x_max = 1.1, x_log = False,
             y_min = 0, y_max = 1.1, y_log = False,
             linestyles=[], colorgrouping=-1,
             extra_lines=[], labels=[],
             atlas_x=-1, atlas_y=-1, simulation=False,
             textlist=[], title=''):
    plt.cla()
    plt.clf()
    
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for extra_line in extra_lines:
        plt.plot(extra_line[0], extra_line[1], linestyle='--', color='black')
        
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (x,y) in enumerate(zip(xlist,ylist)):
        if len(linestyles) > 0:
            linestyle = linestyles[i]
        else:
            linestyle = 'solid'
        if colorgrouping > 0:
            color = colors[int(np.floor(i / colorgrouping))]
        else:
            color = colors[i%(len(colors)-1)]
        label = None
        if len(labels) > 0:
            label = labels[i]
        plt.plot(x, y, label = label, linestyle=linestyle, color=color)
        
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
        
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    ampl.set_xlabel(xlabel)
    ampl.set_ylabel(ylabel)
    
    plt.legend()

    drawLabels(fig, atlas_x, atlas_y, simulation, textlist)
    
    if figfile != '':
        plt.savefig(figfile)
    plt.show()
    
def make_plot(items, figfile = '',
              xlabel = '', ylabel = '',
              x_log = False, y_log = False,
              labels = [], title = '',
             ):
    plt.cla()
    plt.clf()
    
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for i, item in enumerate(items):
        label = None
        if len(labels) >= i:
            label = labels[i]
        plt.plot(item, label=label)
        
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    
    plt.title(title)
    ampl.set_xlabel(xlabel)
    ampl.set_ylabel(ylabel)
    
    plt.legend()
    if figfile != '':
        plt.savefig(figfile)
    plt.show()
 
    
def drawLabels(fig, atlas_x=-1, atlas_y=-1, simulation=False,
               textlist=[]):
    if atlas_x >= 0 and atlas_y >= 0:
        ampl.draw_atlas_label(atlas_x, atlas_y, simulation=simulation, fontsize=18)

    for textdict in textlist:
        fig.axes[0].text(
            textdict['x'], textdict['y'], textdict['text'], 
            transform=fig.axes[0].transAxes, fontsize=18)

def drawLabels_cnn_dev(fig, atlas_x=-1, atlas_y=-1, simulation=False, 
               textlist=[],color='black', desc='', bbox = {}, ax = None, atlas_status='Internal'):
    if atlas_x >= 0 and atlas_y >= 0:
        if len(bbox) > 0:
            ampl.draw_atlas_label(atlas_x, atlas_y, status = atlas_status, ax = ax, simulation=simulation, fontsize=18, color=color, bbox=bbox, desc=desc)
        else:
            ampl.draw_atlas_label(atlas_x, atlas_y, status = atlas_status, ax = ax, simulation=simulation, fontsize=18, color=color, desc=desc)

    for textdict in textlist:
        fig.axes[0].text(
            textdict['x'], textdict['y'], textdict['text'], 
            transform=fig.axes[0].transAxes, fontsize=18,color=color)

        
display_digits = 2


class rocVar:
    def __init__(self,
                 name,  # name of variable as it appears in the root file
                 bins,  # endpoints of bins as a list
                 df,   # dataframe to construct subsets from
                 latex='',  # optional latex to display variable name with
                 vlist=None,  # optional list to append class instance to
                 ):
        self.name = name
        self.bins = bins

        if(latex == ''):
            self.latex = name
        else:
            self.latex = latex

        self.selections = []
        self.labels = []
        for i, point in enumerate(self.bins):
            if(i == 0):
                self.selections.append(df[name] < point)
                self.labels.append(
                    self.latex+'<'+str(round(point, display_digits)))
            else:
                self.selections.append(
                    (df[name] > self.bins[i-1]) & (df[name] < self.bins[i]))
                self.labels.append(str(round(
                    self.bins[i-1], display_digits))+'<'+self.latex+'<'+str(round(point, display_digits)))
                if(i == len(bins)-1):
                    self.selections.append(df[name] > point)
                    self.labels.append(
                        self.latex+'>'+str(round(point, display_digits)))

        if(vlist != None):
            vlist.append(self)


def rocScan(varlist, scan_targets, labels, ylabels, data, plotpath='',
            x_min=0., x_max=1.0, y_min=0.0, y_max=1.0, x_log = False, y_log = False, rejection = False,
            x_label = 'False positive rate', y_label = 'True positive rate',
            linestyles=[], colorgrouping=-1,
            extra_lines=[],
            atlas_x=-1, atlas_y=-1, simulation=False,
            textlist=[]):
    '''
    Creates a set of ROC curve plots by scanning over the specified variables.
    One set is created for each target (neural net score dataset).
    
    varlist: a list of rocVar instances to scan over
    scan_targets: a list of neural net score datasets to use
    labels: a list of target names (strings); must be the same length as scan_targets
    '''

    rocs = buildRocs(varlist, scan_targets, labels, ylabels, data)

    for target_label in labels:
        for v in varlist:
            # prepare matplotlib figure
            plt.cla()
            plt.clf()
            fig = plt.figure()
            fig.patch.set_facecolor('white')
            plt.plot([0, 1], [0, 1], 'k--')

            for label in v.labels:
                # first generate ROC curve
                x = rocs[target_label+label]['x']
                y = rocs[target_label+label]['y']
                var_auc = auc(x, y)
                if not rejection:
                    plt.plot(x, y, label=label+' (area = {:.3f})'.format(var_auc))
                else:
                    plt.plot(y, 1. / x, label=label +
                             ' (area = {:.3f})'.format(var_auc))

            # plt.title('ROC Scan of '+target_label+' over '+v.latex)
            if x_log:
                plt.xscale('log')
            if y_log:
                plt.yscale('log')
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            ampl.set_xlabel(x_label)
            ampl.set_ylabel(y_label)
            plt.legend()

            drawLabels(fig, atlas_x, atlas_y, simulation, textlist)

            if plotpath != '':
                plt.savefig(plotpath+'roc_scan_' +
                            target_label+'_'+v.name+'.pdf')
            plt.show()

def buildRocs(varlist, scan_targets, labels, ylabels, data):
    rocs = {}
    for target, target_label in zip(scan_targets, labels):
        for v in varlist:
            for binning, label in zip(v.selections, v.labels):
                # first generate ROC curve
                x, y, t = roc_curve(
                    ylabels[data.test & binning][:, 1],
                    target[data.test & binning],
                    drop_intermediate=False,
                )

                rocs[target_label + label] = {'x': x, 'y': y}

    return rocs

def roc_plot_scores(scores, ylabels, data, figfile='',
             x_label='False positive rate',
             y_label='True positive rate',
             x_min=0, x_max=1.1, x_log=False,
             y_min=0, y_max=1.1, y_log=False,
             linestyles=[], colorgrouping=-1,
             extra_lines=[], labels=[], rejection = False,
             atlas_x=-1, atlas_y=-1, simulation=False, atlas_status = 'Internal',
             textlist=[], title=''):
    plt.cla()
    plt.clf()

    fig = plt.figure(figsize=(10,10))
    fig.patch.set_facecolor('white')

    gs = gridspec.GridSpec(ncols=1,nrows=2, height_ratios = [3,1], width_ratios = [1])
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0], sharex = ax1)
    for extra_line in extra_lines:
        ax1.plot(extra_line[0], extra_line[1], linestyle='--', color='black')

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    rocs = buildRocsScore(scores, ylabels, labels, data)
    
    ratxs = {}
    ratys = {}
    for i, roc in enumerate(rocs):
        x = rocs[roc]['x']
        y = rocs[roc]['y']
        ratx, raty = build_ratio(rocs['$\mathcal{P}^{EM}_{clus}$']['x'], rocs['$\mathcal{P}^{EM}_{clus}$']['y'], x, y)
        ratxs[i] = ratx
        ratys[i] = raty
        if len(linestyles) > 0:
            linestyle = linestyles[i]
        else:
            linestyle = 'solid'
        if colorgrouping > 0:
            color = colors[int(np.floor((i+1) / colorgrouping))]
        else:
            color = colors[(i+1) % (len(colors)-1)]
        label = None
        if len(labels) > 0:
            label = labels[i]
        if not rejection:
            ax1.plot(x, y, label=label, linestyle=linestyle, color=color, linewidth=3)
        else:
            ax1.plot(x, 1. / y, label=label, linestyle=linestyle, color=color, linewidth=3)

    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
        ax1.set_yscale('log')
        ax2.set_yscale('linear')

    #drawLabels(fig, atlas_x, atlas_y, simulation, textlist, ax = ax1, atlas_status = atlas_status)
       
    drawLabels(fig, atlas_x, atlas_y, simulation, textlist)
    
    ax1.legend(prop={'size': 14}, frameon=False)

    if rejection:
        ax2.set_ylim(0, 15)

    plt.title(title)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ampl.set_xlabel(x_label, ax=ax1)
    ampl.set_ylabel(y_label, ax=ax1)

    ampl.set_ylabel('Ratio to $\mathcal{P}^{EM}_{clus}$', ax=ax2)

    # plt.subplot(414)
    for x, y in zip(ratxs, ratys):
        # print(ratxs[x],ratys[y])
        if colorgrouping > 0:
            color = colors[int(np.floor((x+1) / colorgrouping))]
        else:
            color = colors[(x+1) % (len(colors)-1)]
        if len(linestyles) > 0:
            linestyle = linestyles[x]
        else:
            linestyle = 'solid'
        ax2.plot(ratxs[x], ratys[y], color = color,linewidth=3,linestyle=linestyle)

    if figfile != '':
        plt.savefig(figfile)
    plt.show()


def buildRocsScore(scores, ylabels, labels, data):
    rocs = {}
    for score, label in zip(scores, labels):
        y, x, t = roc_curve( # note y, x: fpr and tpr respectively, which we use as y and x
            ylabels[data.test][:,1],
            score[data.test],
            drop_intermediate=False
        )
        rocs[label] = {'x': x, 'y': y}
    print("ROC curve values: ", rocs)
    return rocs


def build_ratio(x1, y1, x2, y2):

    if len(x1) == len(x2): # if they're the same length, it's easy
        return x1, y2/y1

    #otherwise, we assume x2/y2 are longer
    # so we have to interpolate x1/y1 into the bigger format

    y1_interp = np.interp(x2, x1, y1)

    return x2, y1_interp / y2
    
