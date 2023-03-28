import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns


def read_metrics(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    # tp, tn = 0
    for line in lines:
        line = line[:-1]
        if "Method" in line:
            method = line.split(": ")[-1]
        if "Recall" in line:
            tp = float(re.search(r'\d*\.?\d+', line).group())
        if "Specificity" in line:
            tn = float(re.search(r'\d*\.?\d+', line).group())

    return method, tp, tn


def read_results(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()
    # maa, rot_err, trans_err = 0
    for line in lines:
        line = line[:-1]
        if "Method" in line:
            method = line.split(": ")[-1]
        if "Accuracy" in line:
            maa = re.search(r'\d*\.?\d+', line)
            if maa is not None:
                maa = float(maa.group())
            # maa = float(re.search(r'\d*\.?\d+', line).group())
        if "rotation" in line:
            # rot_err = float(re.search(r'\d*\.?\d+', line).group())
            rot_err = re.search(r'\d*\.?\d+', line)
            if rot_err is not None:
                rot_err = float(rot_err.group())
        if "translation" in line:
            # trans_err = float(re.search(r'\d*\.?\d+', line).group())
            trans_err = re.search(r'\d*\.?\d+', line)
            if trans_err is not None:
                trans_err = float(trans_err.group())
        if "Filtered" in line:
            reduction_perc = re.search(r'\d*\.?\d+', line)
            if reduction_perc is not None:
                reduction_perc = float(reduction_perc.group())
                reduction_perc = round(reduction_perc/100, 3)

    return method, maa, rot_err, trans_err, reduction_perc


def draw_plots(x_axis, y_axis, annotates, title, x_label, y_label, plot_name, save_path):
    fig, ax = plt.subplots(figsize=(14, 6))
    # filter out methods which has y as None
    none_idx = [i for i in range(len(y_axis)) if y_axis[i] is None]
    none_txt = "Remarks: \n"
    for i in none_idx:
        none_txt = none_txt + annotates[i] + ": None value\n"
    x_axis = np.delete(x_axis, none_idx)
    y_axis = np.delete(y_axis, none_idx)
    annotates = np.delete(annotates, none_idx)

    # separate data of using only xy and xy&rgb features
    rgb_idx = [i for i in range(len(annotates)) if 'rgb' in annotates[i]]
    xy_idx = [i for i in range(len(annotates)) if 'rgb' not in annotates[i]]
    x_rgb = x_axis[rgb_idx]
    x_xy = x_axis[xy_idx]
    a_rgb = annotates[rgb_idx]
    y_rgb = y_axis[rgb_idx]
    y_xy = y_axis[xy_idx]
    a_xy = annotates[xy_idx]

    ax.scatter(x_rgb, y_rgb, label="use xy-coordinate and rgb features")
    for i, txt in enumerate(a_rgb):
        ax.annotate(txt, (x_rgb[i], y_rgb[i]))

    ax.scatter(x_xy, y_xy, label="use xy-coordinate features")
    for i, txt in enumerate(a_xy):
        ax.annotate(txt, (x_xy[i], y_xy[i]))
    # ax.set_xlim(xmin=np.min(x_axis), xmax=np.max(x_axis))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.legend(title="Annotation: method_reductionPercentatge", loc='best')
    if len(none_idx) != 0:
        plt.figtext(0, 0, none_txt, fontsize=12)
    plt.savefig(save_path + "/" + plot_name, dpi=300)
    plt.show()


base_path = "../../Dataset/slice"
slice_nums = ["3", "7", "10"]


def draw_scatter_plots():
    # draw the scatter plot for every slice
    for slice_num in slice_nums:
        tps = {}
        tns = {}
        binary_metric_path = base_path + slice_num + "/ML_data/"
        # metric_files = list(pathlib.Path(binary_metric_path).glob('*.txt'))
        for f in (glob.iglob(binary_metric_path + "*.txt")):
            method, tp, tn = read_metrics(f)
            tps[method] = tp
            tns[method] = tn

        maas = {}
        rot_errs = {}
        trans_errs = {}
        reduction_percs = {}
        results_path = base_path + slice_num + "/exmaps_data/"
        # metric_files = list(pathlib.Path(binary_metric_path).glob('*.txt'))
        for f in (glob.iglob(results_path + "*.txt")):
            method, maa, rot_err, trans_err, reduction_perc = read_results(f)
            maas[method] = maa
            rot_errs[method] = rot_err
            trans_errs[method] = trans_err
            reduction_percs[method] = reduction_perc

        x_tp = np.array(list(tps.values()))
        # x_tp = np.round(x_tp, 4)
        methods = list(tps.keys())
        y_maa = np.array([maas[x] for x in methods])
        y_rot_err = np.array([rot_errs[x] for x in methods])
        y_trans_err = np.array([trans_errs[x] for x in methods])
        annotates = np.array([(x + "_" + str(reduction_percs[x])) for x in methods])

        save_path = base_path + slice_num + "/scatter_plots"
        os.makedirs(save_path, exist_ok=True)
        # plot of tp and maa
        draw_plots(x_tp, y_maa, annotates, 'Slice ' + slice_num + ' - TP and MAA', 'TP (recall)',
                   'MAA', 'tp_maa.png', save_path)

        # plot of tn and maa
        # draw_plots(tns, maas, 'TN and MAA', 'TN (Specificity)', 'MAA', 'tn_maa.png')

        # plot of tp and rot err
        draw_plots(x_tp, y_rot_err, annotates, 'Slice ' + slice_num + ' - TP and Rotation Errors',
                   'TP (recall)', 'Rotation Error (degrees)', 'tp_roterr.png', save_path)

        # plot of tn and rot err
        # draw_plots(tns, rot_errs, 'TN and Rotation Errors', 'TN (Specificity)', 'Rotation Error (degrees)', 'tn_roterr.png')

        # plot of tp and trans err
        draw_plots(x_tp, y_trans_err, annotates, 'Slice ' + slice_num + ' - TP and Translation Errors',
                   'TP (recall)', 'Translation Error (meters)', 'tp_transerr.png', save_path)

        # plot of tn and trans err
        # draw_plots(tns, trans_errs, 'TN and Translation Errors', 'TN (Specificity)', 'Translation Error (meters)', 'tn_transerr.png')


def draw_heap_map():
    nn_err_path = "/results/kerasNN_xy_results/"
    msfe_err_path = "/results/MSFENN_xy_results/"
    for slice_num in slice_nums:
        # read the err.csv into dataframes
        data_path = base_path + slice_num + "/exmaps_data"
        nn_path = data_path + nn_err_path + slice_num + "_kerasNN_xy_errors.csv"
        msfe_path = data_path + msfe_err_path + slice_num + "_MSFENN_xy_errors.csv"

        nn_df = pd.read_csv(nn_path)
        msfe_df = pd.read_csv(msfe_path)

        # get outliers of the errors (for now, I try the last 15 of the ordered error list)
        nn_outliers_name = list((nn_df.iloc[-15:])["img_name"])
        msfe_outliers = list((msfe_df.iloc[-15:])["img_name"])
        is_outlier_in_msfe = []
        for name in nn_outliers_name:
            if not (msfe_df["img_name"] == name).any():
                nn_outliers_name = nn_outliers_name.remove(name)
            else:
                if name in msfe_outliers:
                    is_outlier_in_msfe.append(1)
                else:
                    is_outlier_in_msfe.append(0)
        assert(len(nn_outliers_name) == len(is_outlier_in_msfe))

        nn_outliers_id = [int((name.split("_"))[1]) for name in nn_outliers_name]
        is_outlier_in_nn = [1 for i in range(len(nn_outliers_name))]
        outlier_df = pd.DataFrame({"img_name": nn_outliers_id, "nn": is_outlier_in_nn, "msfe": is_outlier_in_msfe})
        outlier_df.set_index('img_name', inplace=True)
        ax = sns.heatmap(outlier_df, cmap='coolwarm')
        ax.set_xlabel("Methods")
        ax.set_ylabel("Img_name")
        ax.set_title("Slice " + slice_num + ": Outliers in Different Methods")
        plt.show()


def main():
    # draw plots for binary metrics and pose evaluation results
    # draw_scatter_plots()
    draw_heap_map()


if __name__ == "__main__":
    main()