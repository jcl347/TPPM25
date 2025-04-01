from argparse import ArgumentParser
import random
import sys
import warnings
import os
import uuid

import pytorch_lightning as pl
import torch

import SpatialFormer as stf

_MODELS = ["SpatialFormer", "mtgnn", "heuristic", "lstm", "bilstm", "lstnet", "linear", "s4"]

_DSETS = [
    "asos",
    "metr-la",
    "pems-bay",
    "exchange",
    "precip",
    "toy2",
    "solar_energy",
    "syn",
    "mnist",
    "cifar",
    "copy",
    "cont_copy",
    "m4",
    "wiki",
    "ettm1",
    "weather",
    "monash",
    "New-Delhi-AOD",
    "station15_onlyPM",
    "station15_only",
    "station15_k3",
    "station15_k3_dataeng",
    "station15_k3_noSat",
    "station15_k3_2",
    "station15_k4",
    "station15_k5",
    "station15_k6",
    "Data510_2_k3",
    "Data510_2_k4",
    "Data510_2_k5",
    "Data510_Cali_test",
    "Data510_Cali_test_correlated",
    "Data510_Cali_1_k0",
    "Data510_Cali_2_k0",
    "Data510_Cali_3_k0",
    "Data510_Cali_8_k0",
    "Data510_Cali_11_k0",
    "Data510_Cali_16_k0",
    "Data510_Cali_7_k0",
    "Data510_Cali_6_k0",
    "Data510_Cali_10_k0",
    "Data510_Cali_23_k0",
    "Data510_Cali_5_k0",
    "Data510_Cali_22_k0",
    "Data510_Cali_13_k0",
    "Data510_Cali_9_k0",
    "Data510_Cali_34_k0",
    "Data510_Cali_21_k0",
    "Data510_Cali_4_k0",
    "Data510_Cali_33_k0",
    "Data510_Cali_12_k0",
    "Data510_Cali_32_k0",
    "Data510_Cali_15_k0",
    "Data510_Cali_20_k0",
    "Data510_Cali_27_k0",
    "Data510_Cali_31_k0",
    "Data510_Cali_38_k0",
    "Data510_Cali_37_k0",
    "Data510_Cali_17_k0",
    "Data510_Cali_30_k0",
    "Data510_Cali_45_k0",
    "Data510_Cali_47_k0",
    "Data510_Cali_54_k0",
    "Data510_Cali_19_k0",
    "Data510_Cali_26_k0",
    "Data510_Cali_18_k0",
    "Data510_Cali_44_k0",
    "Data510_Cali_25_k0",
    "Data510_Cali_39_k0",
    "Data510_Cali_72_k0",
    "Data510_Cali_43_k0",
    "Data510_Cali_53_k0",
    "Data510_Cali_36_k0",
    "Data510_Cali_66_k0",
    "Data510_Cali_61_k0",
    "Data510_Cali_58_k0",
    "Data510_Cali_71_k0",
    "Data510_Cali_52_k0",
    "Data510_Cali_65_k0",
    "Data510_Cali_74_k0",
    "Data510_Cali_64_k0",
    "Data510_Cali_60_k0",
    "Data510_Cali_63_k0",
    "Data510_Cali_70_k0",
    "Data510_Cali_73_k0",
    "Data510_Cali_78_k0",
    "Data510_Cali_77_k0",
    "Data510_Cali_76_k0",
    "Data510_Cali_107_k0",
    "Data510_Cali_88_k0",
    "Data510_Cali_112_k0",
    "Data510_Cali_96_k0",
    "Data510_Cali_102_k0",
    "Data510_Cali_101_k0",
    "Data510_Cali_100_k0",
    "Data510_Cali_111_k0",
    "Data510_Cali_106_k0",
    "Data510_Cali_110_k0",
    "Data510_Cali_105_k0",
    "Data510_Cali_114_k0",
    "Data510_Cali_113_k0",

    "Data510_Cali_1_k1",
    "Data510_Cali_2_k1",
    "Data510_Cali_3_k1",
    "Data510_Cali_8_k1",
    "Data510_Cali_11_k1",
    "Data510_Cali_16_k1",
    "Data510_Cali_7_k1",
    "Data510_Cali_6_k1",
    "Data510_Cali_10_k1",
    "Data510_Cali_23_k1",
    "Data510_Cali_5_k1",
    "Data510_Cali_13_k1",
    "Data510_Cali_9_k1",
    "Data510_Cali_34_k1",
    "Data510_Cali_4_k1",
    "Data510_Cali_12_k1",
    "Data510_Cali_15_k1",
    "Data510_Cali_20_k1",
    "Data510_Cali_27_k1",
    "Data510_Cali_17_k1",
    "Data510_Cali_30_k1",
    "Data510_Cali_36_k1",
    "Data510_Cali_19_k1",
    "Data510_Cali_26_k1",
    "Data510_Cali_18_k1",
    "Data510_Cali_44_k1",
    "Data510_Cali_25_k1",
    "Data510_Cali_39_k1",
    "Data510_Cali_43_k1",
    "Data510_Cali_53_k1",
    "Data510_Cali_36_k1",
    "Data510_Cali_66_k1",
    "Data510_Cali_61_k1",
    "Data510_Cali_58_k1",
    "Data510_Cali_52_k1",
    "Data510_Cali_74_k1",
    "Data510_Cali_64_k1",
    "Data510_Cali_63_k1",
    "Data510_Cali_76_k1",
    "Data510_Cali_88_k1",
    "Data510_Cali_96_k1",
    "Data510_Cali_100_k1",
    "Data510_Cali_111_k1",
    "Data510_Cali_110_k1",
    "Data510_Cali_105_k1",
    "Data510_Cali_114_k1",

    #k2 and k3
    "Data510_Cali_1_k3",
    "Data510_Cali_23_k3",
    "Data510_Cali_52_k3",
    "Data510_Cali_74_k3",
    "Data510_Cali_36_k3",
    "Data510_Cali_27_k3",
    "Data510_Cali_17_k3",
    "Data510_Cali_44_k3",
    "Data510_Cali_64_k3",
    "Data510_Cali_76_k3",
    "Data510_Cali_88_k3",
    "Data510_Cali_105_k3",

    "Data510_Cali_2_k3",
    "Data510_Cali_3_k3",
    "Data510_Cali_8_k3",
    "Data510_Cali_11_k3",
    "Data510_Cali_16_k3",
    "Data510_Cali_7_k3",
    "Data510_Cali_6_k3",
    "Data510_Cali_10_k3",
    "Data510_Cali_5_k3",
    "Data510_Cali_13_k3",
    "Data510_Cali_9_k3",
    "Data510_Cali_4_k3",
    "Data510_Cali_12_k3",
    "Data510_Cali_15_k3",
    "Data510_Cali_20_k3",
    "Data510_Cali_27_k2",
    "Data510_Cali_17_k2",
    "Data510_Cali_30_k3",
    "Data510_Cali_19_k3",
    "Data510_Cali_26_k3",
    "Data510_Cali_18_k3",
    "Data510_Cali_44_k2",
    "Data510_Cali_25_k3",
    "Data510_Cali_39_k3",
    "Data510_Cali_43_k3",
    "Data510_Cali_53_k3",
    "Data510_Cali_66_k3",
    "Data510_Cali_61_k3",
    "Data510_Cali_58_k3",
    "Data510_Cali_64_k2",
    "Data510_Cali_76_k2",
    "Data510_Cali_88_k2",
    "Data510_Cali_96_k3",
    "Data510_Cali_111_k3",
    "Data510_Cali_110_k3",
    "Data510_Cali_105_k2",
    "Data510_Cali_114_k3",

    #K5 experiments
    "Data510_Cali_1_k5",
    "Data510_Cali_2_k5",
    "Data510_Cali_3_k5",
    "Data510_Cali_8_k5",
    "Data510_Cali_11_k5",
    "Data510_Cali_16_k5",
    "Data510_Cali_7_k5",
    "Data510_Cali_6_k5",
    "Data510_Cali_10_k5",
    "Data510_Cali_23_k5",
    "Data510_Cali_5_k5",
    "Data510_Cali_13_k5",
    "Data510_Cali_9_k5",
    "Data510_Cali_4_k5",
    "Data510_Cali_12_k5",
    "Data510_Cali_15_k5",
    "Data510_Cali_20_k5",
    "Data510_Cali_27_k5",
    "Data510_Cali_17_k5",
    "Data510_Cali_30_k5",
    "Data510_Cali_19_k5",
    "Data510_Cali_26_k5",
    "Data510_Cali_18_k5",
    "Data510_Cali_44_k5",
    "Data510_Cali_25_k5",
    "Data510_Cali_39_k5",
    "Data510_Cali_43_k5",
    "Data510_Cali_53_k5",
    "Data510_Cali_36_k5",
    "Data510_Cali_66_k5",
    "Data510_Cali_61_k5",
    "Data510_Cali_58_k5",
    "Data510_Cali_52_k5",
    "Data510_Cali_74_k5",
    "Data510_Cali_64_k5",
    "Data510_Cali_76_k5",
    "Data510_Cali_88_k5",
    "Data510_Cali_96_k5",
    "Data510_Cali_111_k5",
    "Data510_Cali_110_k5",
    "Data510_Cali_105_k5",
    "Data510_Cali_114_k5",
]


def create_parser():
    model = sys.argv[1]
    dset = sys.argv[2]

    # Throw error now before we get confusing parser issues
    assert (
        model in _MODELS
    ), f"Unrecognized model (`{model}`). Options include: {_MODELS}"
    assert dset in _DSETS, f"Unrecognized dset (`{dset}`). Options include: {_DSETS}"

    parser = ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("dset")

    if dset == "precip":
        stf.data.precip.GeoDset.add_cli(parser)
        stf.data.precip.CONUS_Precip.add_cli(parser)
    elif dset == "metr-la" or dset == "pems-bay":
        stf.data.metr_la.METR_LA_Data.add_cli(parser)
    elif dset == "syn":
        stf.data.synthetic.SyntheticData.add_cli(parser)
        stf.data.CSVTorchDset.add_cli(parser)
    elif dset == "mnist":
        stf.data.image_completion.MNISTDset.add_cli(parser)
    elif dset == "cifar":
        stf.data.image_completion.CIFARDset.add_cli(parser)
    elif dset == "copy":
        stf.data.copy_task.CopyTaskDset.add_cli(parser)
    elif dset == "cont_copy":
        stf.data.cont_copy_task.ContCopyTaskDset.add_cli(parser)
    elif dset == "m4":
        stf.data.m4.M4TorchDset.add_cli(parser)
    elif dset == "wiki":
        stf.data.wiki.WikipediaTorchDset.add_cli(parser)
    elif dset == "monash":
        stf.data.monash.MonashDset.add_cli(parser)
    else:
        stf.data.CSVTimeSeries.add_cli(parser)
        stf.data.CSVTorchDset.add_cli(parser)
    stf.data.DataModule.add_cli(parser)

    if model == "lstm":
        stf.lstm_model.LSTM_Forecaster.add_cli(parser)
        stf.callbacks.TeacherForcingAnnealCallback.add_cli(parser)
    elif model == "bilstm":
        stf.bi_lstm_model.bi_LSTM_Forecaster.add_cli(parser)
        stf.callbacks.TeacherForcingAnnealCallback.add_cli(parser)
    elif model == "lstnet":
        stf.lstnet_model.LSTNet_Forecaster.add_cli(parser)
    elif model == "mtgnn":
        stf.mtgnn_model.MTGNN_Forecaster.add_cli(parser)
    elif model == "heuristic":
        stf.heuristic_model.Heuristic_Forecaster.add_cli(parser)
    elif model == "SpatialFormer":
        stf.SpatialFormer_model.SpatialFormer_Forecaster.add_cli(parser)
    elif model == "linear":
        stf.linear_model.Linear_Forecaster.add_cli(parser)
    elif model == "s4":
        stf.s4_model.S4_Forecaster.add_cli(parser)

    stf.callbacks.TimeMaskedLossCallback.add_cli(parser)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_samples", type=int, default=8)
    parser.add_argument("--attn_plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    parser.add_argument("--no_earlystopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--trials", type=int, default=1, help="How many consecutive trials to run"
    )

    if len(sys.argv) > 3 and sys.argv[3] == "-h":
        parser.print_help()
        sys.exit(0)

    return parser


def create_model(config):
    x_dim, yc_dim, yt_dim = None, None, None
    if config.dset == "metr-la":
        x_dim = 2
        yc_dim = 207
        yt_dim = 207
    elif config.dset == "pems-bay":
        x_dim = 2
        yc_dim = 325
        yt_dim = 325
    elif config.dset == "precip":
        x_dim = 2
        yc_dim = 49
        yt_dim = 49
    elif config.dset == "asos":
        x_dim = 6
        yc_dim = 6
        yt_dim = 6
    elif config.dset == "solar_energy":
        x_dim = 6
        yc_dim = 137
        yt_dim = 137
    elif config.dset == "exchange":
        x_dim = 6
        yc_dim = 8
        yt_dim = 8
    elif config.dset == "toy2":
        x_dim = 6
        yc_dim = 20
        yt_dim = 20
    elif config.dset == "syn":
        x_dim = 5
        yc_dim = 20
        yt_dim = 20
    elif config.dset == "mnist":
        x_dim = 1
        yc_dim = 28
        yt_dim = 28
    elif config.dset == "cifar":
        x_dim = 1
        yc_dim = 3
        yt_dim = 3
    elif config.dset == "copy" or config.dset == "cont_copy":
        x_dim = 1
        yc_dim = config.copy_vars
        yt_dim = config.copy_vars
    elif config.dset == "m4":
        x_dim = 4
        yc_dim = 1
        yt_dim = 1
    elif config.dset == "wiki":
        x_dim = 2
        yc_dim = 1
        yt_dim = 1
    elif config.dset == "monash":
        x_dim = 4
        yc_dim = 1
        yt_dim = 1
    elif config.dset == "ettm1":
        x_dim = 4
        yc_dim = 7
        yt_dim = 7
    elif config.dset == "weather":
        x_dim = 3
        yc_dim = 21
        yt_dim = 21
    elif config.dset == "New-Delhi-AOD":
        x_dim = 6
        yc_dim = 494    
        yt_dim = 38
    elif config.dset == "station15_onlyPM":
        x_dim = 6
        yc_dim = 1  
        yt_dim = 1
    elif config.dset == "station15_only":
        x_dim = 6
        yc_dim = 10   
        yt_dim = 10
    elif config.dset == "station15_k3":
        x_dim = 6
        yc_dim = 40   
        yt_dim = 1
    elif config.dset == "station15_k3_dataeng":
        x_dim = 6
        yc_dim = 34   
        yt_dim = 34
    elif config.dset == "station15_k3_noSat":
        x_dim = 6
        yc_dim = 30   
        yt_dim = 30
    elif config.dset == "station15_k3_2":
        x_dim = 6
        yc_dim = 40   
        yt_dim = 40
    elif config.dset == "station15_k4":
        x_dim = 6
        yc_dim = 50   
        yt_dim = 50
    elif config.dset == "station15_k5":
        x_dim = 6
        yc_dim = 60   
        yt_dim = 60
    elif config.dset == "station15_k6":
        x_dim = 6
        yc_dim = 70   
        yt_dim = 70
    elif config.dset == "Data510_2_k3":
        x_dim = 3
        yc_dim = 16  
        yt_dim = 1
    elif config.dset == "Data510_2_k4":
        x_dim = 3
        yc_dim = 20  
        yt_dim = 1
    elif config.dset == "Data510_2_k5":
        x_dim = 3
        yc_dim = 24  
        yt_dim = 1
    elif config.dset == "Data510_Cali_test":
        x_dim = 3
        yc_dim = 80
        # yt_dim = 20
        yt_dim = 1
    elif config.dset == "Data510_Cali_test_correlated":
        x_dim = 3
        yc_dim = 69
        yt_dim = 20


    elif config.dset == "Data510_Cali_1_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_2_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_3_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_8_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_11_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_16_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_7_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_6_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_10_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_23_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_5_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_22_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_13_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_9_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_34_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_21_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_4_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_33_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_12_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_32_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_15_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_20_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_27_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_31_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_38_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_37_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_17_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_30_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_45_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_47_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_54_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_19_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_26_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_18_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_44_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_25_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_39_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_72_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_43_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_53_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_36_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_66_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_61_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_58_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_71_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_52_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_65_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_74_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_64_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_60_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_63_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_70_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_73_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_78_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_77_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_76_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_107_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_88_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_112_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_96_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_102_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_101_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_100_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_111_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_106_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_18_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_110_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_105_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_114_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1
    elif config.dset == "Data510_Cali_113_k0":
        x_dim = 3
        yc_dim = 4
        yt_dim = 1

    # K 1 experiments

    elif config.dset == "Data510_Cali_1_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_2_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_3_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_8_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_11_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_16_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_7_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_6_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_10_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_23_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_5_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_13_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_9_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_34_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_4_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_12_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_15_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_20_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_27_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_17_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_30_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_19_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_26_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_18_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_44_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_25_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_39_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_43_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_53_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_36_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_66_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_61_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_58_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_52_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_74_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_64_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_63_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_76_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_88_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_96_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_100_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_111_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_110_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_105_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1
    elif config.dset == "Data510_Cali_114_k1":
        x_dim = 3
        yc_dim = 8
        yt_dim = 1

    # K = 2 and 3 experiments

    elif config.dset == "Data510_Cali_1_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_23_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_52_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_74_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_36_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_27_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_17_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_44_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_64_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_76_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_88_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_105_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1

    elif config.dset == "Data510_Cali_2_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_3_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_8_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_11_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_16_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_7_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_6_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_10_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_5_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_13_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_9_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_4_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_12_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_15_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_20_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_27_k2":
        x_dim = 3
        yc_dim = 12
        yt_dim = 1
    elif config.dset == "Data510_Cali_17_k2":
        x_dim = 3
        yc_dim = 12
        yt_dim = 1
    elif config.dset == "Data510_Cali_30_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_19_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_26_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_18_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_44_k2":
        x_dim = 3
        yc_dim = 12
        yt_dim = 1
    elif config.dset == "Data510_Cali_25_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_39_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_43_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_53_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_66_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_61_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_58_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_64_k2":
        x_dim = 3
        yc_dim = 12
        yt_dim = 1
    elif config.dset == "Data510_Cali_76_k2":
        x_dim = 3
        yc_dim = 12
        yt_dim = 1
    elif config.dset == "Data510_Cali_88_k2":
        x_dim = 3
        yc_dim = 12
        yt_dim = 1
    elif config.dset == "Data510_Cali_96_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_111_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_110_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1
    elif config.dset == "Data510_Cali_105_k2":
        x_dim = 3
        yc_dim = 12
        yt_dim = 1
    elif config.dset == "Data510_Cali_114_k3":
        x_dim = 3
        yc_dim = 16
        yt_dim = 1

    #K5 experiments
    elif config.dset == "Data510_Cali_1_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_2_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_3_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_8_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_11_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_16_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_7_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_6_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_10_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_23_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_5_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_13_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_9_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_4_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_12_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_15_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_20_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_27_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_17_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_30_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_19_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_26_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_18_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_44_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_25_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_39_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_43_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_53_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_36_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_66_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_61_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_58_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_52_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_74_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_64_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_76_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_88_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_96_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_111_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_110_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_105_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1
    elif config.dset == "Data510_Cali_114_k5":
        x_dim = 3
        yc_dim = 24
        yt_dim = 1


    # Code to handle special condition of lstnet
    if config.model == "lstnet":
        yt_dim = yc_dim


    assert x_dim is not None
    assert yc_dim is not None
    assert yt_dim is not None

    if config.model == "lstm":
        forecaster = stf.lstm_model.LSTM_Forecaster(
            # encoder
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            time_emb_dim=config.time_emb_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p,
            # training
            learning_rate=config.learning_rate,
            teacher_forcing_prob=config.teacher_forcing_start,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            use_revin=config.use_revin,
            linear_shared_weights=config.linear_shared_weights,
            use_seasonal_decomp=config.use_seasonal_decomp,
        )

    elif config.model == "bilstm":
        forecaster = stf.bi_lstm_model.bi_LSTM_Forecaster(
            # encoder
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            time_emb_dim=config.time_emb_dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            dropout_p=config.dropout_p,
            # training
            learning_rate=config.learning_rate,
            teacher_forcing_prob=config.teacher_forcing_start,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            use_revin=config.use_revin,
            linear_shared_weights=config.linear_shared_weights,
            use_seasonal_decomp=config.use_seasonal_decomp,
        )

    elif config.model == "heuristic":
        forecaster = stf.heuristic_model.Heuristic_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            target_points=config.target_points,
            loss=config.loss,
            method=config.method,
        )
    elif config.model == "mtgnn":
        forecaster = stf.mtgnn_model.MTGNN_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            target_points=config.target_points,
            gcn_depth=config.gcn_depth,
            dropout_p=config.dropout_p,
            node_dim=config.node_dim,
            dilation_exponential=config.dilation_exponential,
            conv_channels=config.conv_channels,
            subgraph_size=config.subgraph_size,
            skip_channels=config.skip_channels,
            end_channels=config.end_channels,
            residual_channels=config.residual_channels,
            layers=config.layers,
            propalpha=config.propalpha,
            tanhalpha=config.tanhalpha,
            learning_rate=config.learning_rate,
            kernel_size=config.kernel_size,
            l2_coeff=config.l2_coeff,
            time_emb_dim=config.time_emb_dim,
            loss=config.loss,
            linear_window=config.linear_window,
            linear_shared_weights=config.linear_shared_weights,
            use_seasonal_decomp=config.use_seasonal_decomp,
            use_revin=config.use_revin,
        )
    elif config.model == "lstnet":
        forecaster = stf.lstnet_model.LSTNet_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            hidRNN=config.hidRNN,
            hidCNN=config.hidCNN,
            hidSkip=config.hidSkip,
            CNN_kernel=config.CNN_kernel,
            skip=config.skip,
            dropout_p=config.dropout_p,
            output_fun=config.output_fun,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            use_revin=config.use_revin,
        )
    elif config.model == "SpatialFormer":
        if hasattr(config, "context_points") and hasattr(config, "target_points"):
            max_seq_len = config.context_points + config.target_points
        elif hasattr(config, "max_len"):
            max_seq_len = config.max_len
        else:
            raise ValueError("Undefined max_seq_len")
        forecaster = stf.SpatialFormer_model.SpatialFormer_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            max_seq_len=max_seq_len,
            start_token_len=config.start_token_len,
            attn_factor=config.attn_factor,
            d_model=config.d_model,
            d_queries_keys=config.d_qk,
            d_values=config.d_v,
            n_heads=config.n_heads,
            e_layers=config.enc_layers,
            d_layers=config.dec_layers,
            d_ff=config.d_ff,
            dropout_emb=config.dropout_emb,
            dropout_attn_out=config.dropout_attn_out,
            dropout_attn_matrix=config.dropout_attn_matrix,
            dropout_qkv=config.dropout_qkv,
            dropout_ff=config.dropout_ff,
            pos_emb_type=config.pos_emb_type,
            use_final_norm=not config.no_final_norm,
            global_self_attn=config.global_self_attn,
            local_self_attn=config.local_self_attn,
            global_cross_attn=config.global_cross_attn,
            local_cross_attn=config.local_cross_attn,
            performer_kernel=config.performer_kernel,
            performer_redraw_interval=config.performer_redraw_interval,
            attn_time_windows=config.attn_time_windows,
            use_shifted_time_windows=config.use_shifted_time_windows,
            norm=config.norm,
            activation=config.activation,
            init_lr=config.init_lr,
            base_lr=config.base_lr,
            warmup_steps=config.warmup_steps,
            decay_factor=config.decay_factor,
            initial_downsample_convs=config.initial_downsample_convs,
            intermediate_downsample_convs=config.intermediate_downsample_convs,
            embed_method=config.embed_method,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            class_loss_imp=config.class_loss_imp,
            recon_loss_imp=config.recon_loss_imp,
            time_emb_dim=config.time_emb_dim,
            null_value=config.null_value,
            pad_value=config.pad_value,
            linear_window=config.linear_window,
            use_revin=config.use_revin,
            linear_shared_weights=config.linear_shared_weights,
            use_seasonal_decomp=config.use_seasonal_decomp,
            use_val=not config.no_val,
            use_time=not config.no_time,
            use_space=not config.no_space,
            use_given=not config.no_given,
            recon_mask_skip_all=config.recon_mask_skip_all,
            recon_mask_max_seq_len=config.recon_mask_max_seq_len,
            recon_mask_drop_seq=config.recon_mask_drop_seq,
            recon_mask_drop_standard=config.recon_mask_drop_standard,
            recon_mask_drop_full=config.recon_mask_drop_full,
        )
    elif config.model == "linear":
        forecaster = stf.linear_model.Linear_Forecaster(
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            context_points=config.context_points,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            linear_shared_weights=config.linear_shared_weights,
            use_revin=config.use_revin,
            use_seasonal_decomp=config.use_seasonal_decomp,
        )
    elif config.model == "s4":
        forecaster = stf.s4_model.S4_Forecaster(
            context_points=config.context_points,
            target_points=config.target_points,
            d_state=config.d_state,
            d_model=config.d_model,
            d_x=x_dim,
            d_yc=yc_dim,
            d_yt=yt_dim,
            layers=config.layers,
            time_emb_dim=config.time_emb_dim,
            channels=config.channels,
            dropout_p=config.dropout_p,
            learning_rate=config.learning_rate,
            l2_coeff=config.l2_coeff,
            loss=config.loss,
            linear_window=config.linear_window,
            linear_shared_weights=config.linear_shared_weights,
            use_revin=config.use_revin,
            use_seasonal_decomp=config.use_seasonal_decomp,
        )

    return forecaster


def create_dset(config):
    INV_SCALER = lambda x: x
    SCALER = lambda x: x
    NULL_VAL = None
    PLOT_VAR_IDXS = None
    PLOT_VAR_NAMES = None
    PAD_VAL = None

    if config.dset == "metr-la" or config.dset == "pems-bay":
        if config.dset == "pems-bay":
            assert (
                "pems_bay" in config.data_path
            ), "Make sure to switch to the pems-bay file!"
        data = stf.data.metr_la.METR_LA_Data(config.data_path)
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.metr_la.METR_LA_Torch,
            dataset_kwargs={"data": data},
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = data.inverse_scale
        SCALER = data.scale
        NULL_VAL = 0.0

    elif config.dset == "precip":
        dset = stf.data.precip.GeoDset(dset_dir=config.dset_dir, var="precip")
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.precip.CONUS_Precip,
            dataset_kwargs={
                "dset": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        NULL_VAL = -1.0
    elif config.dset == "syn":
        dset = stf.data.synthetic.SyntheticData(config.data_path)
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
    elif config.dset in ["mnist", "cifar"]:
        if config.dset == "mnist":
            config.target_points = 28 - config.context_points
            datasetCls = stf.data.image_completion.MNISTDset
            PLOT_VAR_IDXS = [18, 24]
            PLOT_VAR_NAMES = ["18th row", "24th row"]
        else:
            config.target_points = 32 * 32 - config.context_points
            datasetCls = stf.data.image_completion.CIFARDset
            PLOT_VAR_IDXS = [0]
            PLOT_VAR_NAMES = ["Reds"]
        DATA_MODULE = stf.data.DataModule(
            datasetCls=datasetCls,
            dataset_kwargs={"context_points": config.context_points},
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
    elif config.dset == "copy":
        # set these manually in case the model needs them
        config.context_points = config.copy_length + int(
            config.copy_include_lags
        )  # seq + lags
        config.target_points = config.copy_length
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.copy_task.CopyTaskDset,
            dataset_kwargs={
                "length": config.copy_length,
                "copy_vars": config.copy_vars,
                "lags": config.copy_lags,
                "mask_prob": config.copy_mask_prob,
                "include_lags": config.copy_include_lags,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
    elif config.dset == "cont_copy":
        # set these manually in case the model needs them
        config.context_points = config.copy_length + int(
            config.copy_include_lags
        )  # seq + lags
        config.target_points = config.copy_length
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.cont_copy_task.ContCopyTaskDset,
            dataset_kwargs={
                "length": config.copy_length,
                "copy_vars": config.copy_vars,
                "lags": config.copy_lags,
                "include_lags": config.copy_include_lags,
                "magnitude_matters": config.copy_mag_matters,
                "freq_shift": config.copy_freq_shift,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
    elif config.dset == "m4":
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.m4.M4TorchDset,
            dataset_kwargs={
                "data_path": config.data_path,
                "resolutions": args.resolutions,
                "max_len": args.max_len,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            collate_fn=stf.data.m4.pad_m4_collate,
            overfit=args.overfit,
        )
        NULL_VAL = -1.0
        PAD_VAL = -1.0

    elif config.dset == "wiki":
        DATA_MODULE = stf.data.DataModule(
            stf.data.wiki.WikipediaTorchDset,
            dataset_kwargs={
                "data_path": config.data_path,
                "forecast_duration": args.forecast_duration,
                "max_len": args.max_len,
            },
            batch_size=args.batch_size,
            workers=args.workers,
            collate_fn=stf.data.wiki.pad_wiki_collate,
            overfit=args.overfit,
        )
        NULL_VAL = -1.0
        PAD_VAL = -1.0
        SCALER = stf.data.wiki.WikipediaTorchDset.scale
        INV_SCALER = stf.data.wiki.WikipediaTorchDset.inverse_scale
    elif config.dset == "monash":
        root_dir = config.root_dir
        DATA_MODULE = stf.data.monash.monash_dloader.make_monash_dmodule(
            root_dir=root_dir,
            max_len=config.max_len,
            include=config.include,
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=config.overfit,
        )
        NULL_VAL = -64.0
        PAD_VAL = -64.0
    elif config.dset == "ettm1":
        target_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        dset = stf.data.CSVTimeSeries(
            data_path=config.data_path,
            target_cols=target_cols,
            ignore_cols=[],
            val_split=4.0 / 20,  # from informer
            test_split=4.0 / 20,  # from informer
            time_col_name="date",
            time_features=["month", "day", "weekday", "hour"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
        # PAD_VAL = -32.0
        PLOT_VAR_NAMES = target_cols
        PLOT_VAR_IDXS = [i for i in range(len(target_cols))]
    elif config.dset == "weather":
        data_path = config.data_path
        dset = stf.data.CSVTimeSeries(
            data_path=config.data_path,
            target_cols=[],
            ignore_cols=[],
            # paper says 7:1:2 split
            val_split=1.0 / 10,
            test_split=2.0 / 10,
            time_col_name="date",
            time_features=["day", "hour", "minute"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
        PLOT_VAR_NAMES = ["OT", "p (mbar)", "raining (s)"]
        PLOT_VAR_IDXS = [20, 0, 15]

    elif config.dset == "New-Delhi-AOD":
        data_path = "./data/Final_Delhi_2019_Data_revised.csv"
        target_cols = ["Imputed_PM25_15", "Imputed_PM25_16", "Imputed_PM25_17", "Imputed_PM25_18", "Imputed_PM25_19", "Imputed_PM25_20", "Imputed_PM25_21", "Imputed_PM25_22",
        "Imputed_PM25_23", "Imputed_PM25_24", "Imputed_PM25_25", "Imputed_PM25_26", "Imputed_PM25_27", "Imputed_PM25_28", "Imputed_PM25_29", "Imputed_PM25_30", "Imputed_PM25_31",
        "Imputed_PM25_32", "Imputed_PM25_33", "Imputed_PM25_34", "Imputed_PM25_35", "Imputed_PM25_36", "Imputed_PM25_37", "Imputed_PM25_38", "Imputed_PM25_39",
        "Imputed_PM25_40", "Imputed_PM25_41", "Imputed_PM25_42", "Imputed_PM25_43", "Imputed_PM25_44", "Imputed_PM25_45", "Imputed_PM25_46", "Imputed_PM25_47",
        "Imputed_PM25_48", "Imputed_PM25_49", "Imputed_PM25_50", "Imputed_PM25_51", "Imputed_PM25_52"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            val_split= 1/12,
            test_split= 1/12,
            time_col_name="Datetime",
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "station15_onlyPM":
        data_path = "./data/Delhi_Station15_only.csv"
        target_cols = ["Imputed_PM25_15"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=[],
            ignore_cols=[],
            val_split= 1/12,
            test_split= 1/12,
            K_fold_cross = True,
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "station15_only":
        data_path = "./data/Delhi_Station15_only.csv"
        target_cols = ["Imputed_PM25_15"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            val_split= 1/12,
            test_split= 1/12,
            K_fold_cross = True,
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "station15_k3":
        data_path = "./data/Delhi_Station15_k3.csv"
        target_cols = ["Imputed_PM25_15"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            val_split= 1/12,
            test_split= 1/12,
            K_fold_cross = True,
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "station15_k3_dataeng":
        data_path = "./data/Delhi_Station15_k3_dataeng.csv"
        target_cols = ["Imputed_PM25_15"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=[],
            ignore_cols=[],
            val_split= 1/12,
            test_split= 1/12,
            K_fold_cross = True,
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "station15_k3_noSat":
        data_path = "./data/Delhi_Station15_k3_noSat.csv"
        target_cols = ["Imputed_PM25_15"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=[],
            ignore_cols=[],
            val_split= 1/12,
            test_split= 1/12,
            K_fold_cross = True,
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "station15_k3_2":
        data_path = "./data/Delhi_Station15_k3.csv"
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=[],
            ignore_cols=[],
            val_split= 1/12,
            test_split= 1/12,
            K_fold_cross = True,
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "station15_k4":
        data_path = "./data/Delhi_Station15_k4.csv"
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=[],
            ignore_cols=[],
            val_split= 1/12,
            test_split= 1/12,
            K_fold_cross = True,
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "station15_k5":
        data_path = "./data/Delhi_Station15_k5.csv"
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=[],
            ignore_cols=[],
            val_split= 1/12,
            test_split= 1/12,
            K_fold_cross = True,
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "station15_k6":
        data_path = "./data/Delhi_Station15_k6.csv"
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=[],
            ignore_cols=[],
            val_split= 1/12,
            test_split= 1/12,
            K_fold_cross = True,
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "Data510_2_k3":
        data_path = "./data/510_2_k3.csv"
        target_cols = ["PM_2"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
            val_split= .1,
            test_split= 22.0/75,
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "Data510_2_k4":
        data_path = "./data/510_2_k4.csv"
        target_cols = ["PM_2"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "Data510_2_k5":
        data_path = "./data/510_2_k5.csv"
        target_cols = ["PM_2"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "Data510_Cali_test":
        data_path = "./data/510_Cali_test.csv"
        target_cols = ["PM_60"]
        # target_cols = ["PM_100", "PM_110", "PM_76", "PM_114", "PM_101", "PM_77", "PM_96", "PM_70", "PM_88", 
        # "PM_78", "PM_60", "PM_111", "PM_73", "PM_36", "PM_63", "PM_52", "PM_64", "PM_25", "PM_107", "PM_65",]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
            val_split = .3,
            test_split = 0,
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "Data510_Cali_test_correlated":
        data_path = "./data/510_Cali_test_correlation.csv"
        target_cols = ["PM_100", "PM_110", "PM_76", "PM_114", "PM_101", "PM_77", "PM_96", "PM_70", "PM_88", 
        "PM_78", "PM_60", "PM_111", "PM_73", "PM_36", "PM_63", "PM_52", "PM_64", "PM_25", "PM_107", "PM_65",]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
            val_split = .3,
            test_split = 0,
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    # K 0 Trials
    elif config.dset == "Data510_Cali_1_k0":
        data_path = "./data/k0/510_Cali_1_k0.csv"
        # target_cols = ["PM_1"]
        target_cols = []
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "Data510_Cali_2_k0":
        data_path = "./data/k0/510_Cali_2_k0.csv"
        target_cols = ["PM_2"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_3_k0":
        data_path = "./data/k0/510_Cali_3_k0.csv"
        target_cols = ["PM_3"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_8_k0":
        data_path = "./data/k0/510_Cali_8_k0.csv"
        target_cols = ["PM_8"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_11_k0":
        data_path = "./data/k0/510_Cali_11_k0.csv"
        target_cols = ["PM_11"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_16_k0":
        data_path = "./data/k0/510_Cali_16_k0.csv"
        target_cols = ["PM_16"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_7_k0":
        data_path = "./data/k0/510_Cali_7_k0.csv"
        target_cols = ["PM_7"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_6_k0":
        data_path = "./data/k0/510_Cali_6_k0.csv"
        target_cols = ["PM_6"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_10_k0":
        data_path = "./data/k0/510_Cali_10_k0.csv"
        target_cols = ["PM_10"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_23_k0":
        data_path = "./data/k0/510_Cali_23_k0.csv"
        target_cols = ["PM_23"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_5_k0":
        data_path = "./data/k0/510_Cali_5_k0.csv"
        target_cols = ["PM_5"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_22_k0":
        data_path = "./data/k0/510_Cali_22_k0.csv"
        target_cols = ["PM_22"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_13_k0":
        data_path = "./data/k0/510_Cali_13_k0.csv"
        target_cols = ["PM_13"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_9_k0":
        data_path = "./data/k0/510_Cali_9_k0.csv"
        target_cols = ["PM_9"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_34_k0":
        data_path = "./data/k0/510_Cali_34_k0.csv"
        target_cols = ["PM_34"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_21_k0":
        data_path = "./data/k0/510_Cali_21_k0.csv"
        target_cols = ["PM_21"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_4_k0":
        data_path = "./data/k0/510_Cali_4_k0.csv"
        target_cols = ["PM_4"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_33_k0":
        data_path = "./data/k0/510_Cali_33_k0.csv"
        target_cols = ["PM_33"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_12_k0":
        data_path = "./data/k0/510_Cali_12_k0.csv"
        target_cols = ["PM_12"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_32_k0":
        data_path = "./data/k0/510_Cali_32_k0.csv"
        target_cols = ["PM_32"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_15_k0":
        data_path = "./data/k0/510_Cali_15_k0.csv"
        target_cols = ["PM_15"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_20_k0":
        data_path = "./data/k0/510_Cali_20_k0.csv"
        target_cols = ["PM_20"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
            # val_split = .1,
            # test_split = .1,

        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_27_k0":
        data_path = "./data/k0/510_Cali_27_k0.csv"
        target_cols = ["PM_27"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_31_k0":
        data_path = "./data/k0/510_Cali_31_k0.csv"
        target_cols = ["PM_31"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_38_k0":
        data_path = "./data/k0/510_Cali_38_k0.csv"
        target_cols = ["PM_38"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_37_k0":
        data_path = "./data/k0/510_Cali_37_k0.csv"
        target_cols = ["PM_37"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_17_k0":
        data_path = "./data/k0/510_Cali_17_k0.csv"
        target_cols = ["PM_17"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_30_k0":
        data_path = "./data/k0/510_Cali_30_k0.csv"
        target_cols = ["PM_30"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_45_k0":
        data_path = "./data/k0/510_Cali_45_k0.csv"
        target_cols = ["PM_45"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_47_k0":
        data_path = "./data/k0/510_Cali_47_k0.csv"
        target_cols = ["PM_47"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_54_k0":
        data_path = "./data/k0/510_Cali_54_k0.csv"
        target_cols = ["PM_54"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_19_k0":
        data_path = "./data/k0/510_Cali_19_k0.csv"
        target_cols = ["PM_19"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_26_k0":
        data_path = "./data/k0/510_Cali_26_k0.csv"
        target_cols = ["PM_26"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_18_k0":
        data_path = "./data/k0/510_Cali_18_k0.csv"
        target_cols = ["PM_18"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_44_k0":
        data_path = "./data/k0/510_Cali_44_k0.csv"
        target_cols = ["PM_44"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_25_k0":
        data_path = "./data/k0/510_Cali_25_k0.csv"
        target_cols = ["PM_25"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_39_k0":
        data_path = "./data/k0/510_Cali_39_k0.csv"
        target_cols = ["PM_39"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_72_k0":
        data_path = "./data/k0/510_Cali_72_k0.csv"
        target_cols = ["PM_72"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_43_k0":
        data_path = "./data/k0/510_Cali_43_k0.csv"
        target_cols = ["PM_43"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_53_k0":
        data_path = "./data/k0/510_Cali_53_k0.csv"
        target_cols = ["PM_53"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_36_k0":
        data_path = "./data/k0/510_Cali_36_k0.csv"
        target_cols = ["PM_36"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_66_k0":
        data_path = "./data/k0/510_Cali_66_k0.csv"
        target_cols = ["PM_66"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_61_k0":
        data_path = "./data/k0/510_Cali_61_k0.csv"
        target_cols = ["PM_61"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_58_k0":
        data_path = "./data/k0/510_Cali_58_k0.csv"
        target_cols = ["PM_58"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_71_k0":
        data_path = "./data/k0/510_Cali_71_k0.csv"
        target_cols = ["PM_71"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_52_k0":
        data_path = "./data/k0/510_Cali_52_k0.csv"
        target_cols = ["PM_52"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_65_k0":
        data_path = "./data/k0/510_Cali_65_k0.csv"
        target_cols = ["PM_65"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_74_k0":
        data_path = "./data/k0/510_Cali_74_k0.csv"
        target_cols = ["PM_74"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_64_k0":
        data_path = "./data/k0/510_Cali_64_k0.csv"
        target_cols = ["PM_64"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_60_k0":
        data_path = "./data/k0/510_Cali_60_k0.csv"
        target_cols = ["PM_60"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_63_k0":
        data_path = "./data/k0/510_Cali_63_k0.csv"
        target_cols = ["PM_63"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_70_k0":
        data_path = "./data/k0/510_Cali_70_k0.csv"
        target_cols = ["PM_70"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_73_k0":
        data_path = "./data/k0/510_Cali_73_k0.csv"
        target_cols = ["PM_73"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_78_k0":
        data_path = "./data/k0/510_Cali_78_k0.csv"
        target_cols = ["PM_78"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_77_k0":
        data_path = "./data/k0/510_Cali_77_k0.csv"
        target_cols = ["PM_77"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_76_k0":
        data_path = "./data/k0/510_Cali_76_k0.csv"
        target_cols = ["PM_76"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_107_k0":
        data_path = "./data/k0/510_Cali_107_k0.csv"
        target_cols = ["PM_107"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_88_k0":
        data_path = "./data/k0/510_Cali_88_k0.csv"
        target_cols = ["PM_88"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_112_k0":
        data_path = "./data/k0/510_Cali_112_k0.csv"
        target_cols = ["PM_112"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_96_k0":
        data_path = "./data/k0/510_Cali_96_k0.csv"
        target_cols = ["PM_96"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_102_k0":
        data_path = "./data/k0/510_Cali_102_k0.csv"
        target_cols = ["PM_102"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_101_k0":
        data_path = "./data/k0/510_Cali_101_k0.csv"
        target_cols = ["PM_101"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_100_k0":
        data_path = "./data/k0/510_Cali_100_k0.csv"
        target_cols = ["PM_100"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_111_k0":
        data_path = "./data/k0/510_Cali_111_k0.csv"
        target_cols = ["PM_111"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_106_k0":
        data_path = "./data/k0/510_Cali_106_k0.csv"
        target_cols = ["PM_106"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_110_k0":
        data_path = "./data/k0/510_Cali_110_k0.csv"
        target_cols = ["PM_110"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_105_k0":
        data_path = "./data/k0/510_Cali_105_k0.csv"
        target_cols = ["PM_105"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_114_k0":
        data_path = "./data/k0/510_Cali_114_k0.csv"
        target_cols = ["PM_114"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_113_k0":
        data_path = "./data/k0/510_Cali_113_k0.csv"
        target_cols = ["PM_113"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    #K 1 Trials
    elif config.dset == "Data510_Cali_1_k1":
        data_path = "./data/k1/510_Cali_1_k1.csv"
        target_cols = ["PM_1"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_36_k1":
        data_path = "./data/k1/510_Cali_36_k1.csv"
        target_cols = ["PM_36"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_2_k1":
        data_path = "./data/k1/510_Cali_2_k1.csv"
        target_cols = ["PM_2"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_3_k1":
        data_path = "./data/k1/510_Cali_3_k1.csv"
        target_cols = ["PM_3"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_8_k1":
        data_path = "./data/k1/510_Cali_8_k1.csv"
        target_cols = ["PM_8"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_11_k1":
        data_path = "./data/k1/510_Cali_11_k1.csv"
        target_cols = ["PM_11"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_16_k1":
        data_path = "./data/k1/510_Cali_16_k1.csv"
        target_cols = ["PM_16"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_7_k1":
        data_path = "./data/k1/510_Cali_7_k1.csv"
        target_cols = ["PM_7"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_6_k1":
        data_path = "./data/k1/510_Cali_6_k1.csv"
        target_cols = ["PM_6"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_10_k1":
        data_path = "./data/k1/510_Cali_10_k1.csv"
        target_cols = ["PM_10"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_23_k1":
        data_path = "./data/k1/510_Cali_23_k1.csv"
        target_cols = ["PM_23"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_5_k1":
        data_path = "./data/k1/510_Cali_5_k1.csv"
        target_cols = ["PM_5"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_13_k1":
        data_path = "./data/k1/510_Cali_13_k1.csv"
        target_cols = ["PM_13"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_9_k1":
        data_path = "./data/k1/510_Cali_9_k1.csv"
        target_cols = ["PM_9"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_34_k1":
        data_path = "./data/k1/510_Cali_34_k1.csv"
        target_cols = ["PM_34"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_4_k1":
        data_path = "./data/k1/510_Cali_4_k1.csv"
        target_cols = ["PM_4"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_12_k1":
        data_path = "./data/k1/510_Cali_12_k1.csv"
        target_cols = ["PM_12"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_15_k1":
        data_path = "./data/k1/510_Cali_15_k1.csv"
        target_cols = ["PM_15"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_20_k1":
        data_path = "./data/k1/510_Cali_20_k1.csv"
        target_cols = ["PM_20"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_27_k1":
        data_path = "./data/k1/510_Cali_27_k1.csv"
        target_cols = ["PM_27"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_17_k1":
        data_path = "./data/k1/510_Cali_17_k1.csv"
        target_cols = ["PM_17"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_30_k1":
        data_path = "./data/k1/510_Cali_30_k1.csv"
        target_cols = ["PM_30"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_19_k1":
        data_path = "./data/k1/510_Cali_19_k1.csv"
        target_cols = ["PM_19"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_26_k1":
        data_path = "./data/k1/510_Cali_26_k1.csv"
        target_cols = ["PM_26"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_18_k1":
        data_path = "./data/k1/510_Cali_18_k1.csv"
        target_cols = ["PM_18"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_44_k1":
        data_path = "./data/k1/510_Cali_44_k1.csv"
        target_cols = ["PM_44"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_25_k1":
        data_path = "./data/k1/510_Cali_25_k1.csv"
        target_cols = ["PM_25"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_39_k1":
        data_path = "./data/k1/510_Cali_39_k1.csv"
        target_cols = ["PM_39"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_43_k1":
        data_path = "./data/k1/510_Cali_43_k1.csv"
        target_cols = ["PM_43"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_25_k1":
        data_path = "./data/k1/510_Cali_25_k1.csv"
        target_cols = ["PM_25"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_39_k1":
        data_path = "./data/k1/510_Cali_39_k1.csv"
        target_cols = ["PM_39"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_43_k1":
        data_path = "./data/k1/510_Cali_43_k1.csv"
        target_cols = ["PM_43"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_53_k1":
        data_path = "./data/k1/510_Cali_53_k1.csv"
        target_cols = ["PM_53"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_36_k1":
        data_path = "./data/k1/510_Cali_36_k1.csv"
        target_cols = ["PM_36"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_66_k1":
        data_path = "./data/k1/510_Cali_66_k1.csv"
        target_cols = ["PM_66"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_61_k1":
        data_path = "./data/k1/510_Cali_61_k1.csv"
        target_cols = ["PM_61"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_58_k1":
        data_path = "./data/k1/510_Cali_58_k1.csv"
        target_cols = ["PM_58"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_52_k1":
        data_path = "./data/k1/510_Cali_52_k1.csv"
        target_cols = ["PM_52"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_36_k1":
        data_path = "./data/k1/510_Cali_36_k1.csv"
        target_cols = ["PM_36"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_66_k1":
        data_path = "./data/k1/510_Cali_66_k1.csv"
        target_cols = ["PM_66"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_61_k1":
        data_path = "./data/k1/510_Cali_61_k1.csv"
        target_cols = ["PM_61"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_58_k1":
        data_path = "./data/k1/510_Cali_58_k1.csv"
        target_cols = ["PM_58"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_52_k1":
        data_path = "./data/k1/510_Cali_52_k1.csv"
        target_cols = ["PM_52"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_74_k1":
        data_path = "./data/k1/510_Cali_74_k1.csv"
        target_cols = ["PM_74"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_64_k1":
        data_path = "./data/k1/510_Cali_64_k1.csv"
        target_cols = ["PM_64"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_63_k1":
        data_path = "./data/k1/510_Cali_63_k1.csv"
        target_cols = ["PM_63"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_76_k1":
        data_path = "./data/k1/510_Cali_76_k1.csv"
        target_cols = ["PM_76"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_88_k1":
        data_path = "./data/k1/510_Cali_88_k1.csv"
        target_cols = ["PM_88"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_96_k1":
        data_path = "./data/k1/510_Cali_96_k1.csv"
        target_cols = ["PM_96"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_100_k1":
        data_path = "./data/k1/510_Cali_100_k1.csv"
        target_cols = ["PM_100"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_111_k1":
        data_path = "./data/k1/510_Cali_111_k1.csv"
        target_cols = ["PM_111"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_110_k1":
        data_path = "./data/k1/510_Cali_110_k1.csv"
        target_cols = ["PM_110"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_105_k1":
        data_path = "./data/k1/510_Cali_105_k1.csv"
        target_cols = ["PM_105"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_114_k1":
        data_path = "./data/k1/510_Cali_114_k1.csv"
        target_cols = ["PM_114"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    #k2 and k3 trials
    elif config.dset == "Data510_Cali_1_k3":
        data_path = "./data/k3/510_Cali_1_k3.csv"
        target_cols = ["PM_1"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_23_k3":
        data_path = "./data/k3/510_Cali_23_k3.csv"
        target_cols = ["PM_23"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_52_k3":
        data_path = "./data/k3/510_Cali_52_k3.csv"
        target_cols = ["PM_52"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_74_k3":
        data_path = "./data/k3/510_Cali_74_k3.csv"
        target_cols = ["PM_74"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_36_k3":
        data_path = "./data/k3/510_Cali_36_k3.csv"
        target_cols = ["PM_36"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_27_k3":
        data_path = "./data/k3/510_Cali_27_k3.csv"
        target_cols = ["PM_27"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_17_k3":
        data_path = "./data/k3/510_Cali_17_k3.csv"
        target_cols = ["PM_17"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_44_k3":
        data_path = "./data/k3/510_Cali_44_k3.csv"
        target_cols = ["PM_44"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_64_k3":
        data_path = "./data/k3/510_Cali_64_k3.csv"
        target_cols = ["PM_64"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_76_k3":
        data_path = "./data/k3/510_Cali_76_k3.csv"
        target_cols = ["PM_76"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_88_k3":
        data_path = "./data/k3/510_Cali_88_k3.csv"
        target_cols = ["PM_88"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_105_k3":
        data_path = "./data/k3/510_Cali_105_k3.csv"
        target_cols = ["PM_105"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_2_k3":
        data_path = "./data/k3/510_Cali_2_k3.csv"
        target_cols = ["PM_2"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_3_k3":
        data_path = "./data/k3/510_Cali_3_k3.csv"
        target_cols = ["PM_3"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_8_k3":
        data_path = "./data/k3/510_Cali_8_k3.csv"
        target_cols = ["PM_8"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_11_k3":
        data_path = "./data/k3/510_Cali_11_k3.csv"
        target_cols = ["PM_11"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_16_k3":
        data_path = "./data/k3/510_Cali_16_k3.csv"
        target_cols = ["PM_16"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_7_k3":
        data_path = "./data/k3/510_Cali_7_k3.csv"
        target_cols = ["PM_7"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_6_k3":
        data_path = "./data/k3/510_Cali_6_k3.csv"
        target_cols = ["PM_6"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_10_k3":
        data_path = "./data/k3/510_Cali_10_k3.csv"
        target_cols = ["PM_10"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_5_k3":
        data_path = "./data/k3/510_Cali_5_k3.csv"
        target_cols = ["PM_5"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_13_k3":
        data_path = "./data/k3/510_Cali_13_k3.csv"
        target_cols = ["PM_13"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_9_k3":
        data_path = "./data/k3/510_Cali_9_k3.csv"
        target_cols = ["PM_9"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_4_k3":
        data_path = "./data/k3/510_Cali_4_k3.csv"
        target_cols = ["PM_4"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_25_k3":
        data_path = "./data/k3/510_Cali_25_k3.csv"
        target_cols = ["PM_25"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_12_k3":
        data_path = "./data/k3/510_Cali_12_k3.csv"
        target_cols = ["PM_12"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_15_k3":
        data_path = "./data/k3/510_Cali_15_k3.csv"
        target_cols = ["PM_15"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_20_k3":
        data_path = "./data/k3/510_Cali_20_k3.csv"
        target_cols = ["PM_20"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_27_k2":
        data_path = "./data/k3/510_Cali_27_k2.csv"
        target_cols = ["PM_27"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_17_k2":
        data_path = "./data/k3/510_Cali_17_k2.csv"
        target_cols = ["PM_17"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_30_k3":
        data_path = "./data/k3/510_Cali_30_k3.csv"
        target_cols = ["PM_30"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_19_k3":
        data_path = "./data/k3/510_Cali_19_k3.csv"
        target_cols = ["PM_19"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_26_k3":
        data_path = "./data/k3/510_Cali_26_k3.csv"
        target_cols = ["PM_26"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_18_k3":
        data_path = "./data/k3/510_Cali_18_k3.csv"
        target_cols = ["PM_18"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_44_k2":
        data_path = "./data/k3/510_Cali_44_k2.csv"
        target_cols = ["PM_44"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_25_k2":
        data_path = "./data/k3/510_Cali_25_k3.csv"
        target_cols = ["PM_25"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_39_k3":
        data_path = "./data/k3/510_Cali_39_k3.csv"
        target_cols = ["PM_39"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_43_k3":
        data_path = "./data/k3/510_Cali_43_k3.csv"
        target_cols = ["PM_43"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_43_k3":
        data_path = "./data/k3/510_Cali_43_k3.csv"
        target_cols = ["PM_43"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_53_k3":
        data_path = "./data/k3/510_Cali_53_k3.csv"
        target_cols = ["PM_53"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_66_k3":
        data_path = "./data/k3/510_Cali_66_k3.csv"
        target_cols = ["PM_66"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_61_k3":
        data_path = "./data/k3/510_Cali_61_k3.csv"
        target_cols = ["PM_61"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_53_k3":
        data_path = "./data/k3/510_Cali_53_k3.csv"
        target_cols = ["PM_58"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    elif config.dset == "Data510_Cali_66_k3":
        data_path = "./data/k3/510_Cali_66_k3.csv"
        target_cols = ["PM_66"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_61_k3":
        data_path = "./data/k3/510_Cali_61_k3.csv"
        target_cols = ["PM_61"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_58_k3":
        data_path = "./data/k3/510_Cali_58_k3.csv"
        target_cols = ["PM_58"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_64_k2":
        data_path = "./data/k3/510_Cali_64_k2.csv"
        target_cols = ["PM_64"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_76_k2":
        data_path = "./data/k3/510_Cali_76_k2.csv"
        target_cols = ["PM_76"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_88_k2":
        data_path = "./data/k3/510_Cali_88_k2.csv"
        target_cols = ["PM_88"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_96_k3":
        data_path = "./data/k3/510_Cali_96_k3.csv"
        target_cols = ["PM_96"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_111_k3":
        data_path = "./data/k3/510_Cali_111_k3.csv"
        target_cols = ["PM_111"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_110_k3":
        data_path = "./data/k3/510_Cali_110_k3.csv"
        target_cols = ["PM_110"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_105_k2":
        data_path = "./data/k3/510_Cali_105_k2.csv"
        target_cols = ["PM_105"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_114_k3":
        data_path = "./data/k3/510_Cali_114_k3.csv"
        target_cols = ["PM_114"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    # K5 Experiments

    elif config.dset == "Data510_Cali_1_k5":
        data_path = "./data/k5/510_Cali_1_k5.csv"
        target_cols = ["PM_1"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_2_k5":
        data_path = "./data/k5/510_Cali_2_k5.csv"
        target_cols = ["PM_2"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_3_k5":
        data_path = "./data/k5/510_Cali_3_k5.csv"
        target_cols = ["PM_3"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_8_k5":
        data_path = "./data/k5/510_Cali_8_k5.csv"
        target_cols = ["PM_8"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_11_k5":
        data_path = "./data/k5/510_Cali_11_k5.csv"
        target_cols = ["PM_11"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_16_k5":
        data_path = "./data/k5/510_Cali_16_k5.csv"
        target_cols = ["PM_16"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_7_k5":
        data_path = "./data/k5/510_Cali_7_k5.csv"
        target_cols = ["PM_7"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_6_k5":
        data_path = "./data/k5/510_Cali_6_k5.csv"
        target_cols = ["PM_6"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_10_k5":
        data_path = "./data/k5/510_Cali_10_k5.csv"
        target_cols = ["PM_10"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_23_k5":
        data_path = "./data/k5/510_Cali_23_k5.csv"
        target_cols = ["PM_23"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_5_k5":
        data_path = "./data/k5/510_Cali_5_k5.csv"
        target_cols = ["PM_5"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_13_k5":
        data_path = "./data/k5/510_Cali_13_k5.csv"
        target_cols = ["PM_13"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_9_k5":
        data_path = "./data/k5/510_Cali_9_k5.csv"
        target_cols = ["PM_9"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_4_k5":
        data_path = "./data/k5/510_Cali_4_k5.csv"
        target_cols = ["PM_4"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_12_k5":
        data_path = "./data/k5/510_Cali_12_k5.csv"
        target_cols = ["PM_12"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_15_k5":
        data_path = "./data/k5/510_Cali_15_k5.csv"
        target_cols = ["PM_15"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_20_k5":
        data_path = "./data/k5/510_Cali_20_k5.csv"
        target_cols = ["PM_20"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_27_k5":
        data_path = "./data/k5/510_Cali_27_k5.csv"
        target_cols = ["PM_27"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_17_k5":
        data_path = "./data/k5/510_Cali_17_k5.csv"
        target_cols = ["PM_17"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_30_k5":
        data_path = "./data/k5/510_Cali_30_k5.csv"
        target_cols = ["PM_30"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_19_k5":
        data_path = "./data/k5/510_Cali_19_k5.csv"
        target_cols = ["PM_19"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_26_k5":
        data_path = "./data/k5/510_Cali_26_k5.csv"
        target_cols = ["PM_26"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_18_k5":
        data_path = "./data/k5/510_Cali_18_k5.csv"
        target_cols = ["PM_18"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_44_k5":
        data_path = "./data/k5/510_Cali_44_k5.csv"
        target_cols = ["PM_44"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_25_k5":
        data_path = "./data/k5/510_Cali_25_k5.csv"
        target_cols = ["PM_25"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_39_k5":
        data_path = "./data/k5/510_Cali_39_k5.csv"
        target_cols = ["PM_39"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_43_k5":
        data_path = "./data/k5/510_Cali_43_k5.csv"
        target_cols = ["PM_43"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_53_k5":
        data_path = "./data/k5/510_Cali_53_k5.csv"
        target_cols = ["PM_53"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_36_k5":
        data_path = "./data/k5/510_Cali_36_k5.csv"
        target_cols = ["PM_36"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_66_k5":
        data_path = "./data/k5/510_Cali_66_k5.csv"
        target_cols = ["PM_66"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_61_k5":
        data_path = "./data/k5/510_Cali_61_k5.csv"
        target_cols = ["PM_61"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_58_k5":
        data_path = "./data/k5/510_Cali_58_k5.csv"
        target_cols = ["PM_58"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_52_k5":
        data_path = "./data/k5/510_Cali_52_k5.csv"
        target_cols = ["PM_52"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_74_k5":
        data_path = "./data/k5/510_Cali_74_k5.csv"
        target_cols = ["PM_74"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_64_k5":
        data_path = "./data/k5/510_Cali_64_k5.csv"
        target_cols = ["PM_64"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_76_k5":
        data_path = "./data/k5/510_Cali_76_k5.csv"
        target_cols = ["PM_76"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_88_k5":
        data_path = "./data/k5/510_Cali_88_k5.csv"
        target_cols = ["PM_88"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_96_k5":
        data_path = "./data/k5/510_Cali_96_k5.csv"
        target_cols = ["PM_96"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_111_k5":
        data_path = "./data/k5/510_Cali_111_k5.csv"
        target_cols = ["PM_111"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_110_k5":
        data_path = "./data/k5/510_Cali_110_k5.csv"
        target_cols = ["PM_110"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_105_k5":
        data_path = "./data/k5/510_Cali_105_k5.csv"
        target_cols = ["PM_105"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None
    elif config.dset == "Data510_Cali_114_k5":
        data_path = "./data/k5/510_Cali_114_k5.csv"
        target_cols = ["PM_114"]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols=[],
            time_features = ["year", "month", "day"],
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

    else:
        data_path = config.data_path
        if config.dset == "asos":
            if data_path == "auto":
                data_path = "./data/temperature-v1.csv"
            target_cols = ["ABI", "AMA", "ACT", "ALB", "JFK", "LGA"]
        elif config.dset == "solar_energy":
            if data_path == "auto":
                data_path = "./data/solar_AL_converted.csv"
            target_cols = [str(i) for i in range(137)]
        elif "toy" in config.dset:
            if data_path == "auto":
                if config.dset == "toy2":
                    data_path = "./data/toy_dset2.csv"
                else:
                    raise ValueError(f"Unrecognized toy dataset {config.dset}")
            target_cols = [f"D{i}" for i in range(1, 21)]
        elif config.dset == "exchange":
            if data_path == "auto":
                data_path = "./data/exchange_rate_converted.csv"
            target_cols = [
                "Australia",
                "United Kingdom",
                "Canada",
                "Switzerland",
                "China",
                "Japan",
                "New Zealand",
                "Singapore",
            ]
        dset = stf.data.CSVTimeSeries(
            data_path=data_path,
            target_cols=target_cols,
            ignore_cols="all",
        )
        DATA_MODULE = stf.data.DataModule(
            datasetCls=stf.data.CSVTorchDset,
            dataset_kwargs={
                "csv_time_series": dset,
                "context_points": config.context_points,
                "target_points": config.target_points,
                "time_resolution": config.time_resolution,
            },
            batch_size=config.batch_size,
            workers=config.workers,
            overfit=args.overfit,
        )
        INV_SCALER = dset.reverse_scaling
        SCALER = dset.apply_scaling
        NULL_VAL = None

        if config.model == "lstnet":
            target_cols = []

    return (
        DATA_MODULE,
        INV_SCALER,
        SCALER,
        NULL_VAL,
        PLOT_VAR_IDXS,
        PLOT_VAR_NAMES,
        PAD_VAL,
    )


def create_callbacks(config, save_dir):
    filename = f"{config.run_name}_" + str(uuid.uuid1()).split("-")[0]
    model_ckpt_dir = os.path.join(save_dir, filename)
    config.model_ckpt_dir = model_ckpt_dir
    saving = pl.callbacks.ModelCheckpoint(
        dirpath=model_ckpt_dir,
        monitor="val/loss",
        mode="min",
        filename=f"{config.run_name}" + "{epoch:02d}",
        save_top_k=1,
        auto_insert_metric_name=True,
    )
    callbacks = [saving]

    if not config.no_earlystopping:
        callbacks.append(
            pl.callbacks.early_stopping.EarlyStopping(
                monitor="val/loss",
                patience=config.patience,
            )
        )

    if config.wandb:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    if config.model == "lstm":
        callbacks.append(
            stf.callbacks.TeacherForcingAnnealCallback(
                start=config.teacher_forcing_start,
                end=config.teacher_forcing_end,
                steps=config.teacher_forcing_anneal_steps,
            )
        )
    if config.time_mask_loss:
        callbacks.append(
            stf.callbacks.TimeMaskedLossCallback(
                start=config.time_mask_start,
                end=config.time_mask_end,
                steps=config.time_mask_anneal_steps,
            )
        )
    return callbacks


def main(args):
    log_dir = os.getenv("STF_LOG_DIR")
    if log_dir is None:
        log_dir = "./data/STF_LOG_DIR"
        print(
            "Using default wandb log dir path of ./data/STF_LOG_DIR. This can be adjusted with the environment variable `STF_LOG_DIR`"
        )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if args.wandb:
        import wandb

        project = "SpatialFormer"
        entity = "jl29293"
        assert (
            project is not None and entity is not None
        ), "Please set environment variables `STF_WANDB_ACCT` and `STF_WANDB_PROJ` with \n\
            your wandb user/organization name and project title, respectively."
        experiment = wandb.init(
            project=project,
            entity=entity,
            config=args,
            dir=log_dir,
            reinit=True,
        )
        config = wandb.config
        wandb.run.name = args.run_name
        wandb.run.save()
        logger = pl.loggers.WandbLogger(
            experiment=experiment,
            save_dir=log_dir,
        )

    # Dset
    (
        data_module,
        inv_scaler,
        scaler,
        null_val,
        plot_var_idxs,
        plot_var_names,
        pad_val,
    ) = create_dset(args)

    # Model
    args.null_value = null_val
    args.pad_value = pad_val
    forecaster = create_model(args)
    forecaster.set_inv_scaler(inv_scaler)
    forecaster.set_scaler(scaler)
    forecaster.set_null_value(null_val)

    # Callbacks
    callbacks = create_callbacks(args, save_dir=log_dir)
    test_samples = next(iter(data_module.test_dataloader()))

    if args.wandb and args.plot:
        callbacks.append(
            stf.plot.PredictionPlotterCallback(
                test_samples,
                var_idxs=plot_var_idxs,
                var_names=plot_var_names,
                pad_val=pad_val,
                total_samples=min(args.plot_samples, args.batch_size),
            )
        )

    if args.wandb and args.dset in ["mnist", "cifar"] and args.plot:
        callbacks.append(
            stf.plot.ImageCompletionCallback(
                test_samples,
                total_samples=min(16, args.batch_size),
                mode="left-right" if config.dset == "mnist" else "flat",
            )
        )

    if args.wandb and args.dset == "copy" and args.plot:
        callbacks.append(
            stf.plot.CopyTaskCallback(
                test_samples,
                total_samples=min(16, args.batch_size),
            )
        )

    if args.wandb and args.model == "SpatialFormer" and args.attn_plot:

        callbacks.append(
            stf.plot.AttentionMatrixCallback(
                test_samples,
                layer=0,
                total_samples=min(1, args.batch_size),
            )
        )

    if args.wandb:
        config.update(args)
        logger.log_hyperparams(config)

    if args.val_check_interval <= 1.0:
        val_control = {"val_check_interval": args.val_check_interval}
    else:
        val_control = {"check_val_every_n_epoch": int(args.val_check_interval)}

    trainer = pl.Trainer(
        gpus=args.gpus,
        callbacks=callbacks,
        logger=logger if args.wandb else None,
        accelerator="cuda",
        gradient_clip_val=args.grad_clip_norm,
        gradient_clip_algorithm="norm",
        overfit_batches=20 if args.debug else 0,
        accumulate_grad_batches=args.accumulate,
        sync_batchnorm=True,
        limit_val_batches=args.limit_val_batches,
        **val_control,
    )
    
        

    # Train
    trainer.fit(forecaster, datamodule=data_module)

    # Test
    trainer.test(datamodule=data_module, ckpt_path="best")

    # Predict (only here as a demo and test)
    # forecaster.to("cuda")
    # xc, yc, xt, _ = test_samples
    # yt_pred = forecaster.predict(xc, yc, xt)

    if args.wandb:
        experiment.finish()


if __name__ == "__main__":
    # CLI
    parser = create_parser()
    args = parser.parse_args()

    for trial in range(args.trials):
        main(args)
