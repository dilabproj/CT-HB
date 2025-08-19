import numpy as np
import pandas as pd
from sklearn import metrics
from scipy import stats
from pycm  import *


PATH_DICT = {
    # 'proposed': {
    #     'path_root': './models_saved/20200508-172733_downstream_non-linear_mitbih_MSwKM_causalcnn',
    #     'path': [
    #         'model_92b8ca917a5740be9cda1644a00e2ebb_Epoch_53_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_620d8fe980004375a4895599f5fe0f9c_Epoch_77_max-valid-Accuracy_2_final_pred_result.csv',
    #         'model_2515662428134da9a631347364768ffc_Epoch_90_max-valid-Accuracy_3_final_pred_result.csv'
    #     ]
    # },
    # 'proposed_0.8': {
    #     'path_root': './models_saved/low_data_test/mitbih/non_linear/20200510-222710_downstream_non-linear_mitbih_MSwKM_causalcnn_0.8',
    #     'path': [
    #         'model_1c451a4771994062b7cd960bbf805a1c_Epoch_31_max-valid-Accuracy_2_final_pred_result.csv',
    #         'model_5d8b721f07e64f7b977f182d35990a02_Epoch_62_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_ee3d44944ded49668c3928b5d2b8bf77_Epoch_57_max-valid-Accuracy_1_final_pred_result.csv'
    #     ]
    # },
    # 'proposed_0.5': {
    #     'path_root': './models_saved/low_data_test/mitbih/non_linear/20200525-141423_downstream_non-linear_mitbih_MSwKM_causalcnn_0.5',
    #     'path': [
    #         'model_8ab8ef1627c74f3dae016e1a24a4fa8b_Epoch_47_max-valid-Accuracy_2_final_pred_result.csv',
    #         'model_3079a15e32d14085ade011c74e959c72_Epoch_45_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_29f0b162e60c44d1b0a2532ad9d43853_Epoch_56_max-valid-Accuracy_3_final_pred_result.csv'
    #     ]
    # },
    # 'proposed_0.1': {
    #     'path_root': './models_saved/low_data_test/mitbih/non_linear/20200525-151235_downstream_non-linear_mitbih_MSwKM_causalcnn_0.1',
    #     'path': [
    #         'model_ec7e95966c06488cbeca18706d761a55_Epoch_63_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_77660a86a1b942e99634c3e125e986ad_Epoch_81_max-valid-Accuracy_2_final_pred_result.csv',
    #         'model_36b36c6078384e76a56ca21e1477c330_Epoch_39_max-valid-Accuracy_3_final_pred_result.csv'
    #     ]
    # },
    # 'proposed_0.05': {
    #     'path_root': './models_saved/low_data_test/mitbih/non_linear/20200511-112208_downstream_non-linear_mitbih_MSwKM_causalcnn_0.05',
    #     'path': [
    #         'model_f497302fc1804ae19934db6393ccc850_Epoch_63_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_cdd8a2fadf1c43a381095f47bbafda5d_Epoch_88_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_0eb88a95c8074d7eab9232bbe2c6cdd0_Epoch_62_max-valid-Accuracy_2_final_pred_result.csv'
    #     ]
    # },
    # 'proposed_v2': {
    #     'path_root': './models_saved/20200524-154652_downstream_non-linear_mitbih_MSwKM_causalcnn',
    #     'path': [
    #         'model_fdf80af4a84f4f83bc0911358036c856_Epoch_42_max-valid-Accuracy_2_final_pred_result.csv',
    #         'model_c425ab00d5414b5ebb942abe1d9e7daf_Epoch_30_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_0e4aad91d7a64731bc1990ac599bf635_Epoch_45_max-valid-Accuracy_1_final_pred_result.csv'
    #     ]
    # },
    # 'proposed_tri': {
    #     'path_root': './models_saved/20200509-111854_downstream_non-linear_mitbih_MSwKM_causalcnn',
    #     'path': [
    #         'model_c66d089864f0460e89fe94f9e61a081a_Epoch_90_max-valid-Accuracy_2_final_pred_result.csv',
    #         'model_b9a732aac5984a38bbd66f26565e778f_Epoch_49_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_a3917b4ec7b24829bb245a8f68bc11d7_Epoch_34_max-valid-Accuracy_1_final_pred_result.csv'
    #     ]
    # },
    # 'proposed_ft': {
    #     'path_root': './models_saved/20200525-012457_downstream_non-linear_mitbih_MSwKM_causalcnn',
    #     'path': [
    #         'model_9d0ce25531d649288a429848145ba2f8_Epoch_81_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_7711b73a27af431a9ee4c84b677f9768_Epoch_97_max-valid-Accuracy_2_final_pred_result.csv',
    #         'model_3b666cb79aba485abb28f77c3f6c8d27_Epoch_92_max-valid-Accuracy_1_final_pred_result.csv'
    #     ]
    # },
    # 'proposed_ft_0.8': {
    #     'path_root': './models_saved/20200522-193019_downstream_non-linear_mitbih_MSwKM_causalcnn',
    #     'path': [
    #         'model_e187009ae1fd400b942b88efa498e8a2_Epoch_47_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_86f3c38af5a9452a9d9d69227b0e0645_Epoch_35_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_73921ca33ef6437c8ddb487b35db2ae7_Epoch_49_max-valid-Accuracy_2_final_pred_result.csv'
    #     ]
    # },
    # 'proposed_ft_0.5': {
    #     'path_root': './models_saved/20200525-160529_downstream_non-linear_mitbih_MSwKM_causalcnn',
    #     'path': [
    #         'model_aeb3754b873249e6afe8f23aff54a27b_Epoch_87_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_a28bd069481c4c41b2f0a2c47156b126_Epoch_92_max-valid-Accuracy_2_final_pred_result.csv',
    #         'model_0c61e622b3984323888438b8b61563a5_Epoch_96_max-valid-Accuracy_3_final_pred_result.csv'
    #     ]
    # },
    # 'proposed_ft_0.1': {
    #     'path_root': './models_saved/20200525-182145_downstream_non-linear_mitbih_MSwKM_causalcnn_0.1',
    #     'path': [
    #         'model_e0c85d7649334267bd607740434e7ed1_Epoch_97_max-valid-Accuracy_2_final_pred_result.csv',
    #         'model_bf8a9f4d844045058751ff7df1669982_Epoch_96_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_84a0212aff2742b987435e7a54e4bb9d_Epoch_99_max-valid-Accuracy_3_final_pred_result.csv'
    #     ]
    # },
    # 'proposed_ft_0.05': {
    #     'path_root': './models_saved/20200511-112208_downstream_non-linear_mitbih_MSwKM_causalcnn_0.05',
    #     'path': [
    #         'model_f497302fc1804ae19934db6393ccc850_Epoch_63_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_cdd8a2fadf1c43a381095f47bbafda5d_Epoch_88_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_0eb88a95c8074d7eab9232bbe2c6cdd0_Epoch_62_max-valid-Accuracy_2_final_pred_result.csv'
    #     ]
    # },
    # 'proposed_trans': {
    #     'path_root': './models_saved/20200510-164407_downstream_non-linear_lvh_MSwKM_causalcnn',
    #     'path': [
    #         'model_b8e29603e9204d2186266d4350da23c5_Epoch_31_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_9cdc643eb2d84bb6a047180967fb6bb0_Epoch_68_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_236e1531defb47759323b72690bc193a_Epoch_77_max-valid-Accuracy_2_final_pred_result.csv'
    #     ]
    # },
    # 'proposed_160': {
    #     'path_root': './models_saved/20200513-145548_downstream_non-linear_mitbih_MSwKM_causalcnn_160',
    #     'path': [
    #         'model_2b61a811aed544a68b9375a6e2f95cb5_Epoch_100_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_3fbe48c33ae74643a74ffb4f11178ee0_Epoch_8_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_7bd4192d4445409a8ffba9c9cd303a74_Epoch_97_max-valid-Accuracy_2_final_pred_result.csv'
    #     ]
    # },
    # 'proposed_320': {
    #     'path_root': './models_saved/20200515-114825_downstream_non-linear_mitbih_MSwKM_causalcnn',
    #     'path': [
    #         'model_c4baaed6c58b42c5bdd3e50f7001c0e9_Epoch_74_max-valid-Accuracy_2_final_pred_result.csv',
    #         'model_92f227a8fdb444188a3021881b0ef303_Epoch_73_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_8a25620c900a4a3499dde7a338a59472_Epoch_71_max-valid-Accuracy_1_final_pred_result.csv'
    #     ]
    # },
    # 'proposed_640': {
    #     'path_root': './models_saved/20200513-175247_downstream_non-linear_mitbih_MSwKM_causalcnn_640',
    #     'path': [
    #         'model_dd43011dd7cb412a8d6da47091e1ae84_Epoch_92_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_8582bd9e5558418d82bdc52a83e75a55_Epoch_84_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_342a4a06f2704b83a119b0700f98fc1d_Epoch_94_max-valid-Accuracy_2_final_pred_result.csv'
    #     ]
    # },
    # 'proposed_1280': {
    #     'path_root': './models_saved/20200514-181723_downstream_non-linear_mitbih_MSwKM_causalcnn',
    #     'path': [
    #         'model_a2c5e5d189814da6a10056707309b1d1_Epoch_71_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_9665ad60faa44d3f9d6396f934441796_Epoch_91_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_68921e42ff45416fb4daba62fb26560e_Epoch_73_max-valid-Accuracy_2_final_pred_result.csv'
    #     ]
    # },
    # 'supervised': {
    #     'path_root': './models_saved/20200427-232518_mitbih_supervised',
    #     'path': [
    #         'model_c4c155a22c8f436688160e92d54b7d9c_Epoch_92_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_9ba591995a4f48d2ae54e6ad847135ff_Epoch_75_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_29ac036cee5f4b779b9d1ace448ee5c1_Epoch_93_max-valid-Accuracy_2_final_pred_result.csv'
    #     ]
    # },
    # 'supervised_0.8': {
    #     'path_root': './models_saved/low_data_test/mitbih/supervised/20200428-132902_mitbih_supervised_0.8',
    #     'path': [
    #         'model_dd62b8dcb81e41eab8a5eb69952f1c4c_Epoch_97_max-valid-Accuracy_2_final_pred_result.csv',
    #         'model_ba0ac933224b44ea946cad8f068e5c82_Epoch_90_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_676d30fb836f4ea1948f4acface85172_Epoch_96_max-valid-Accuracy_1_final_pred_result.csv'
    #     ]
    # },
    # 'supervised_0.5': {
    #     'path_root': './models_saved/low_data_test/mitbih/supervised/20200523-094036_mitbih_supervised_0.5',
    #     'path': [
    #         'model_e6d4daad5202447bb1d1936a7d1a55c8_Epoch_81_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_7bc39c7089a949bf9b483079b526ebfe_Epoch_87_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_2874c6b82e65448fbf98b9ac4c1739b5_Epoch_97_max-valid-Accuracy_2_final_pred_result.csv'
    #     ]
    # },
    # 'supervised_0.1': {
    #     'path_root': './models_saved/low_data_test/mitbih/supervised/20200525-131530_mitbih_supervised_0.1',
    #     'path': [
    #         'model_ef6b55585fe64b839e2673f8a1faca21_Epoch_95_max-valid-Accuracy_2_final_pred_result.csv',
    #         'model_cb8a1c7947264dcdbd217edeb895a093_Epoch_66_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_bc37c909434846bab4bb97118b26fae5_Epoch_89_max-valid-Accuracy_3_final_pred_result.csv'
    #     ]
    # },
    # 'supervised_0.05': {
    #     'path_root': './models_saved/low_data_test/mitbih/supervised/20200428-113215_mitbih_supervised_0.05',
    #     'path': [
    #         'model_cd1abcca9fc04e1aa22282e20133b798_Epoch_63_max-valid-Accuracy_2_final_pred_result.csv',
    #         'model_78feb0a65b9c43bfbb52ca49d2b81e41_Epoch_51_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_00b48695098847cbafa7d0a17e1f9b5a_Epoch_76_max-valid-Accuracy_1_final_pred_result.csv'
    #     ]
    # },
    # 'unsup_mts': {
    #     'path_root': './models_saved/20200427-191023_downstream_non-linear_mitbih_unsup_mts_causalcnn',
    #     'path': [
    #         'model_cff992b87bde463ca6e8a2c302700f98_Epoch_59_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_1b73b3b07d0d42f4bdd4bb752e3b0938_Epoch_72_max-valid-Accuracy_2_final_pred_result.csv',
    #         'model_07f8d01d1ad44eccb0c2658361a82e10_Epoch_43_max-valid-Accuracy_1_final_pred_result.csv'
    #     ]
    # },
    # 'unsup_mts_trans': {
    #     'path_root': './models_saved/20200521-201652_downstream_non-linear_lvh_unsup_mts_causalcnn',
    #     'path': [
    #         'model_cc0ad16a667046348edfb2b20eac8500_Epoch_39_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_63e57b97d353431dac64a05b4ad20273_Epoch_58_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_295fa0a8ef5e4ba68d49aebf00cc0fc6_Epoch_61_max-valid-Accuracy_2_final_pred_result.csv'
    #     ]
    # },
    # 'cdae': {
    #     'path_root': './models_saved/20200520-173025_downstream_non-linear_mitbih_cdae',
    #     'path': [
    #         'model_f850402e8d7247d18e0b736bb55538d9_Epoch_85_max-valid-Accuracy_2_final_pred_result.csv',
    #         'model_bf1a5cb9162a439cb7feedd99cdd22d8_Epoch_58_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_254211059caa4a39b1b2e316f1cada71_Epoch_76_max-valid-Accuracy_3_final_pred_result.csv'
    #     ]
    # },
    # 'dae': {
    #     'path_root': './models_saved/20200520-164036_downstream_non-linear_mitbih_dae',
    #     'path': [
    #         'model_e53048dc3e9a417c8b7b3a4cb442456c_Epoch_98_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_bb7d2d24e4204d81a02d8b7b79d623e5_Epoch_74_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_7ae2757c355b498bbe2f4461611aa391_Epoch_89_max-valid-Accuracy_2_final_pred_result.csv'
    #     ]
    # },
    # 'emotion': {
    #     'path_root': './models_saved/20200424-200451_downstream_non-linear_mitbih_emotion_ssl_original',
    #     'path': [
    #         'model_7a12be5bec9b42ddb82f396f34e90141_Epoch_48_max-valid-Accuracy_1_final_pred_result.csv',
    #         'model_26c510cf9c0f4ce8b1b96954eeebf9e6_Epoch_69_max-valid-Accuracy_3_final_pred_result.csv',
    #         'model_1378e2bc0f4a4ed8b5e903f437d6e9e3_Epoch_79_max-valid-Accuracy_2_final_pred_result.csv'
    #     ]
    # },
}


if __name__ == "__main__":
    for key, _ in PATH_DICT.items():
        path_root = PATH_DICT[key]['path_root']
        pred_prob_list, pred_list, mcc_list = [], [], []
        sveb_acc_list, veb_acc_list, sveb_auc_list, veb_auc_list = [], [], [], []
        tpr_thres = 0.05
        for path in PATH_DICT[key]['path']:
            df = pd.read_csv(f'{path_root}/{path}')
            y = np.array(df['truth'])
            pred_prob_list.append(df['pred_prob'])
            pred_list.append(df['pred'])
            mcc = metrics.matthews_corrcoef(y, df['pred'])
            mcc_list.append(mcc)
            cm = ConfusionMatrix(np.array(y), np.array(df['pred']))
            sveb_acc_list.append(cm.ACC[1])
            veb_acc_list.append(cm.ACC[2])
            sveb_auc_list.append(cm.AUC[1])
            veb_auc_list.append(cm.AUC[2])
        # y = np.array(df['truth'])
        average_MCC_score = np.average(np.array(mcc_list), axis=0)
        average_sveb_acc_score = np.average(np.array(sveb_acc_list), axis=0)
        average_veb_acc_score = np.average(np.array(veb_acc_list), axis=0)
        average_sveb_auc_score = np.average(np.array(sveb_auc_list), axis=0)
        average_veb_auc_score = np.average(np.array(veb_auc_list), axis=0)
        # avg_pred_prob = np.average(np.array(pred_prob_list), axis=0)
        # avg_pred = stats.mode(np.array(pred_list), axis=0)[0].reshape(-1)
        # average_cohen_kappa_score = metrics.cohen_kappa_score(y, avg_pred)
        # average_MCC_score = metrics.matthews_corrcoef(y, avg_pred)
        # print(f'{key} / Average kappa score: {average_cohen_kappa_score}')
        print(f'{key} / Average MCC score: {average_MCC_score}')
        # print(f'{key} / Average SVEB ACC score: {average_sveb_acc_score}')
        # print(f'{key} / Average SVEB AUC score: {average_sveb_auc_score}')
        # print(f'{key} / Average VEB ACC score: {average_veb_acc_score}')
        # print(f'{key} / Average VEB AUC score: {average_veb_auc_score}')
