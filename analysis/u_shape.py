import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

file_0_noise = "./csvs/" + "results_agent_with_noise_0_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
file_0_25_noise = "./csvs/" + "results_agent_with_noise_0.25_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
file_0_5_noise_1000_units = "./csvs/" + "results_agent_with_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
file_0_5_noise_1500_units = "./csvs/" + "results_agent_with_noise_0.5_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1500_units_3_windowsize_0.9_discount_factor" + ".csv"
file_0_5_noise_2000_units = "./csvs/" + "results_agent_with_noise_0.5_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_2000_units_3_windowsize_0.9_discount_factor" + ".csv"
file_0_75_noise = "./csvs/" + "results_agent_with_noise_0.75_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"
file_1_noise = "./csvs/" + "results_agent_with_noise_1_random_emergence_ar_no_mem_ppo_3_act_1e-05_lr_1000_units_3_windowsize_0.9_discount_factor" + ".csv"


# confidence interval stuff
x=np.array([0,0.25,0.5,0.75,1])
means_att = []
means_att_random_schema = []
means_catch = []
means_catch_random_schema = []
noise_files = [file_0_noise, file_0_25_noise, file_0_5_noise_1000_units, file_0_75_noise, file_1_noise]
for f in range(len(noise_files)):
    file = pd.read_csv(noise_files[f])
    means_att.append(np.mean(file["attention_score"][-40:]))
    means_catch.append(np.mean(file["catch_score"][-40:]))
    means_att_random_schema.append(np.mean(file["attention_score_random_schema"][-40:]))
    means_catch_random_schema.append(np.mean(file["catch_score_random_schema"][-40:]))

x=np.array([int(0),0.25,0.5,0.75,int(1)])
means_att_ci = []
means_att_random_schema_ci = []
means_catch_ci = []
means_catch_random_schema_ci = []
for idx, noise in enumerate(x):
    if noise == 0 or noise ==1:
        noise = int(noise)
    file1 = np.array(pd.read_csv("./csvs/test_8_random_acts/" + "not_random_noise_" +str(noise)+ ".csv"))
    file2 = np.array(pd.read_csv("./csvs/test_8_random_acts/" + "3_actions_noise_" +str(noise)+ ".csv"))
    means_att_ci.append(np.mean(file1[1:,0]))
    means_catch_ci.append(np.mean(file1[1:,1]))
    means_att_random_schema_ci.append(np.mean(file2[1:, 0]))
    means_catch_random_schema_ci.append(np.mean(file2[1:,1]))

print("means_att_ci")
print(means_att_ci)
print("means_att_random_schema_ci")
print(means_att_random_schema_ci)
print("means_catch_ci")
print(means_catch_ci)
print("means_catch_random_schema_ci")
print(means_catch_random_schema_ci)

plt.figure()
plt.bar(x-0.025, means_att, width=0.05, label="Learned Attention Sschema")
plt.bar(x+0.025, means_att_random_schema, width=0.05, color="r", label="Random Attention Schema")
plt.xticks(x, fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel("Noise Level", fontsize = 13)
plt.ylabel("Attention Tracking Reward", fontsize = 13)
plt.ylim(0,5)

plt.legend(fontsize = 13)
plt.savefig("./plots/u-shape-att.png")

plt.figure()
plt.bar(x-0.025, means_catch, width=0.05, label="Learned Attention Schema")
plt.bar(x+0.025, means_catch_random_schema, width=0.05, color="r", label="Random Attention Schema")
plt.xticks(x, fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel("Noise Level", fontsize = 13)
plt.ylabel("Catching Ball Reward", fontsize = 13)
plt.ylim(0,2.5)

plt.legend(fontsize = 13)
plt.savefig("./plots/u-shape-catch.png")

# confidence intervalls - tr
x=np.array([0,0.25,0.5,0.75,1])
# means_normal = np.array([[3.7585, 1.98],[3.5745, 1.616],[3.7445, 1.726],[3.7405, 1.73],[3.976, 1.97]]) +np.array([0,0.5])
# means_random_schema = np.array([[2.1105, 1.534],[1.1605, 0.236],[0.93, -0.03],[1.568 ,0.312],[3.6935, 1.656]]) +np.array([0,0.5])
# means_att = means_normal[:,0]
# means_att_random_schema = means_random_schema[:,0]
# means_catch = means_normal[:,1]
# means_catch_random_schema = means_random_schema[:,1]
yerr_normal_att = [0.0875, 0.104, 0.0885, 0.0945, 0.0275]
yerr_random_att = [0.196, 0.2325, 0.246, 0.2435, 0.094]
yerr_normal_catch = [0.026, 0.104, 0.088, 0.09, 0.03]
yerr_random_catch = [0.114,0.17, 0.174,0.17, 0.098]

plt.figure()
plt.bar(x-0.025, means_att_ci, width=0.05, label="Learned Additional Resource", yerr=yerr_normal_att,  capsize=3,bottom=0)
plt.bar(x+0.025, means_att_random_schema_ci, width=0.05, color="r", label="Random Additional Resource", yerr=yerr_random_att,  capsize=3, bottom=0)
plt.xticks(x, fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel("Noise Probability", fontsize = 13)
plt.ylabel("Ball Tracking Reward (TR)", fontsize = 13)
plt.ylim(0,5.1)
plt.axhline(y=4, color='gray', linestyle='--')
plt.text(0.15, 4.02, 'max', color='gray', fontsize=12, ha='center', va='bottom')

plt.legend(fontsize = 13)
plt.savefig("./plots/u-shape-att-ci.png")

# confidence intervalls -cr
bottom = -0.5
means_catch_ci = np.array(means_catch) - bottom
means_catch_random_schema_ci = np.array(means_catch_random_schema) - bottom
plt.figure()
plt.bar(x-0.025, means_catch_ci, width=0.05, label="Learned Additional Resource", yerr=yerr_normal_catch,  capsize=3, bottom=bottom)
plt.bar(x+0.025, means_catch_random_schema_ci, width=0.05, color="r", label="Random Additional Resource", yerr=yerr_random_catch,  capsize=3,bottom=bottom)
plt.xticks(x, fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel("Noise Probability", fontsize = 13)
plt.ylabel("Catching Ball Reward (CR)", fontsize = 13)
plt.ylim(-0.5,2.7)
plt.axhline(y=2, color='gray', linestyle='--')
plt.text(0.15, 2.03, 'max', color='gray', fontsize=12, ha='center', va='bottom')


plt.legend(fontsize = 13)
plt.savefig("./plots/u-shape-catch-ci.png")

# for decoding ball position
performance_vis_input = np.array([1,0,0.777,0,1])
performance_vis_input_schema_pos = np.array([1,0,0.9075,0,1])

plt.figure()
plt.bar(x-0.025, performance_vis_input_schema_pos, width=0.05, label="Visual Input and AS Position")
plt.bar(x+0.025, performance_vis_input, width=0.05, color="r", label="Only Visual Input")
plt.xticks(x, fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel("Noise Level", fontsize = 13)
plt.ylabel("Decoding Accuracy", fontsize = 13)
plt.ylim(0.5,1.1)
plt.legend(fontsize = 13)
plt.savefig("./plots/u-shape-decoding.png")

# for decoding ball position normal vs random as
performance_vis_input_schema_pos_2 = np.array([1,0.9382,0.9295,0.948,1])
performance_random_vis_input_schema_pos = np.array([0.9565,0.9259,0.89,0.9205,1])

plt.figure()
plt.bar(x-0.025, performance_vis_input_schema_pos_2, width=0.05, label="Visual Input and Attention Schema Input")
plt.bar(x+0.025, performance_random_vis_input_schema_pos, width=0.05, color="r", label="Visual Input and Random Attention Schema Input")
plt.xticks(x, fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel("Noise Level", fontsize = 13)
plt.ylabel("Decoding Accuracy", fontsize = 13)
plt.ylim(0.5,1.1)
plt.legend(fontsize = 13)
plt.savefig("./plots/u-shape-decoding-ball-vs-random.png")

# differences decoding ball (normal v random)
aw_decoding_differences = performance_vis_input_schema_pos_2 - performance_random_vis_input_schema_pos
catching_differences = np.array(means_catch) - np.array(means_catch_random_schema)
att_differences = np.array(means_att) - np.array(means_att_random_schema)

plt.figure()
plt.scatter(aw_decoding_differences, att_differences, s=50, label="Attention Tracking Reward")
plt.scatter(aw_decoding_differences, catching_differences, s=50, color="r", label="Catching Ball Reward")
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel("Differences in Decoding Ball Position Accuracy", fontsize = 13)
plt.ylabel("Differences in Rewards", fontsize = 13)
#plt.ylim(0,1.1)
plt.legend(fontsize = 13)

plt.savefig("./plots/u-shape-differences-ball-vs-random.png")


# for decoding attention window


aw_performance_vis_input = np.array([0.6595,0.343,0.6005,0.8587,1])
aw_performance_vis_input_schema_pos = np.array([0.745,0.6835,0.85,0.9402,1]
)
aw_performance_schema = np.array([0.5385,0.5330,0.6120,0.5596,0.5220])

cv_aw_performance_vis_input = np.array([0.6768,0.3397,0.5965,0.8417,0.9995])
cv_aw_performance_vis_input_schema_pos = np.array([0.6932,0.6862,0.8078,0.9287,0.9999])
cv_aw_performance_from_schema = np.array([0.5542,0.5323,0.6111,0.5772,0.5096])
cv_aw_performance_from_random_schema = np.array([0.3438,0.3500,0.4122,0.3884,0.3662])
cv_aw_performance_from_random_schema = np.array([0.3436,0.3487,0.3969,0.3753,0.3599])
#aw_performance_vis_input_schema_col = np.array([0.732,0.43,0.678,0.87,1])
lala = np.array([0.5595, 0.5274, 0.5512 ,0.5681, 0.5045])
lala_random = np.array([0.3495, 0.3330,0.4395,0.4070, 0.3585])
aw_performance_random_schema = np.array([0.5745, 0.5851, 0.7958, 0.9205, 1])

plt.figure()
plt.bar(x-0.025, aw_performance_vis_input_schema_pos, width=0.05, label="Visual Input and Additional Resource")
#plt.bar(x+0.0, aw_performance_vis_input_schema_col, width=0.05, color="orange", label="Visual Input and Additional Resource Column")
plt.bar(x+0.025, aw_performance_vis_input, width=0.05, color="r", label="Only Visual Input")

plt.xticks(x, fontsize = 13)
plt.xlabel("Noise Probability", fontsize = 13)
plt.ylabel(" Attention Window Location Classification Accuracy", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylim(0,1.4)
plt.legend(fontsize = 13)
plt.savefig("./plots/u-shape-decoding_aw.png")

#decoding from schema
plt.figure()
plt.bar(x-0.025, lala, width=0.05, label="Only Additional Resource")
plt.bar(x+0.025, lala_random, width=0.05, color="r", label="Only Random Additional Resource")

plt.xticks(x, fontsize = 13)
plt.xlabel("Noise Probability", fontsize = 13)
plt.ylabel(" Attention Window Location Classification Accuracy", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylim(0,0.8)
plt.legend(fontsize = 13)
plt.savefig("./plots/u-shape-decoding_aw-from_schema.png")

#for random schema
plt.figure()
plt.bar(x-0.025, aw_performance_vis_input_schema_pos, width=0.05, label="Visual Input and Attention Schema Input")
plt.bar(x+0.025, aw_performance_random_schema, width=0.05, color="r", label="Visual Input and Random Attention Schema Input")
plt.xticks(x, fontsize = 13)
plt.xlabel("Noise Level", fontsize = 13)
plt.ylabel("Accuracy of Decoding Attention Window Position", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylim(0,1.2)
plt.legend(fontsize = 13)
plt.savefig("./plots/u-shape-decoding_aw_vs_random.png")

##cv results
#decoding from schema
plt.figure()
plt.bar(x-0.025, cv_aw_performance_from_schema, width=0.05, label="Only Additional Resource")
plt.bar(x+0.025, cv_aw_performance_from_random_schema, width=0.05, color="r", label="Only Random Additional Resource")

plt.xticks(x, fontsize = 13)
plt.xlabel("Noise Probability", fontsize = 13)
plt.ylabel(" Attention Window Location Classification Accuracy", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylim(0,0.8)
plt.legend(fontsize = 13)
plt.savefig("./plots/u-shape-cv-decoding_aw-from_schema.png")

#for random schema
plt.figure()
plt.bar(x, cv_aw_performance_from_schema - cv_aw_performance_from_random_schema, width=0.05, label="Diff")

plt.xticks(x, fontsize = 13)
plt.xlabel("Noise Probability", fontsize = 13)
plt.ylabel(" Attention Window Location Classification Accuracy", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylim(0,0.4)
plt.legend(fontsize = 13)
plt.savefig("./plots/u-shape-cv-decoding_aw-from_schema_differences.png")

#decoding differences add res and random add res
plt.figure()
plt.bar(x-0.025, cv_aw_performance_vis_input_schema_pos, width=0.05, label="Visual Input and Attention Schema Input")
plt.bar(x+0.025, cv_aw_performance_vis_input, width=0.05, color="r", label="Only Visual Input")

plt.xticks(x, fontsize = 13)
plt.xlabel("Noise Probability", fontsize = 13)
plt.ylabel(" Attention Window Location Classification Accuracy", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylim(0,1.3)
plt.legend(fontsize = 13)
plt.savefig("./plots/u-shape-cv-decoding_aw.png")

# decoding aw column from diff things



cv_aw_col_performance_vis_input = np.array([0.63,0.3397,0.52,0.76,1])
cv_aw_col_performance_vis_input_schema_pos = np.array([0.76,0.66,0.7,0.89,1])
cv_aw_col_performance_from_schema = np.array([0.57,0.52,0.61,0.56,0.53])
cv_aw_col_performance_from_random_schema = np.array([0.35,0.3500,0.45,0.38,0.37])


##cv results
#decoding from schema vs random schema
plt.figure()
plt.bar(x-0.025, cv_aw_col_performance_from_schema, width=0.05, label="Only Additional Resource")
plt.bar(x+0.025, cv_aw_col_performance_from_random_schema, width=0.05, color="r", label="Only Random Additional Resource")

plt.xticks(x, fontsize = 13)
plt.xlabel("Noise Probability", fontsize = 13)
plt.ylabel(" Attention Window Column Classification Accuracy", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylim(0,0.8)
plt.legend(fontsize = 13)
plt.savefig("./plots/u-shape-cv-decoding_aw_col-from_schema.png")

#decoding from whole input vs from vis input only
plt.figure()
plt.bar(x-0.025, cv_aw_col_performance_vis_input_schema_pos, width=0.05, label="Visual Input and Attention Schema Input")
plt.bar(x+0.025, cv_aw_col_performance_vis_input, width=0.05, color="r", label="Only Visual Input")

plt.xticks(x, fontsize = 13)
plt.xlabel("Noise Probability", fontsize = 13)
plt.ylabel(" Attention Window Column Classification Accuracy", fontsize = 13)
plt.yticks(fontsize = 13)
plt.ylim(0,1.3)
plt.legend(fontsize = 13)
plt.savefig("./plots/u-shape-cv-decoding_aw_col.png")





aw_decoding_differences = aw_performance_vis_input_schema_pos - aw_performance_vis_input
catching_differences = np.array(means_catch) - np.array(means_catch_random_schema)
att_differences = np.array(means_att) - np.array(means_att_random_schema)
print(aw_decoding_differences)
arr = np.zeros((5,2))
arr[:,0] = aw_decoding_differences
arr[:,1] = att_differences
print(arr)
arr[:,1] = catching_differences

print(arr)
plt.figure()
plt.scatter(aw_decoding_differences, att_differences, s=50, label="Ball Tracking Reward (TR)")
plt.scatter(aw_decoding_differences, catching_differences, s=50, color="r", label="Catching Ball Reward (CR)")
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel("Difficulty of Attention State Inference (ΔAccuracy)", fontsize = 13)
plt.ylabel("Usefulness of Additional Resource (ΔReward)", fontsize = 13)
#plt.ylim(0,1.1)
plt.legend(fontsize = 13)


plt.savefig("./plots/u-shape-differences.png")

#for differences as - random as
aw_decoding_differences = aw_performance_vis_input_schema_pos - aw_performance_random_schema
catching_differences = np.array(means_catch) - np.array(means_catch_random_schema)
att_differences = np.array(means_att) - np.array(means_att_random_schema)

plt.figure()
plt.scatter(aw_decoding_differences, att_differences, s=50, label="Attention Tracking Reward")
plt.scatter(aw_decoding_differences, catching_differences, s=50, color="r", label="Catching Ball Reward")
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.xlabel("Differences in Decoding AW Position Accuracy", fontsize = 13)
plt.ylabel("Differences in Rewards", fontsize = 13)
#plt.ylim(0,1.1)
plt.legend(fontsize = 13)


plt.savefig("./plots/u-shape-differences-random.png")


# diff units

x=np.array([1000,1500,2000])
means_att = []
means_att_random_schema = []
means_catch = []
means_catch_random_schema = []
noise_files = [file_0_5_noise_1000_units, file_0_5_noise_1500_units, file_0_5_noise_2000_units]
for f in range(len(noise_files)):
    file = pd.read_csv(noise_files[f])
    means_att.append(np.mean(file["attention_score"][-10:]))
    means_catch.append(np.mean(file["catch_score"][-10:]))
    means_att_random_schema.append(np.mean(file["attention_score_random_schema"][-10:]))
    means_catch_random_schema.append(np.mean(file["catch_score_random_schema"][-10:]))

plt.figure()
plt.bar(x-25, means_att, width=50, label="Learned AS")
plt.bar(x+25, means_att_random_schema, width=50, color="r", label="Random AS")
plt.xticks(x)
plt.xlabel("Units per Layer")
plt.ylabel("Attention Tracking Reward")
plt.ylim(0,5)

plt.legend()
plt.savefig("./plots/units-u-shape-att.png")

plt.figure()
plt.bar(x-25, means_catch, width=50, label="Learned AS")
plt.bar(x+25, means_catch_random_schema, width=50, color="r", label="Random AS")
plt.xticks(x)
plt.xlabel("Units per Layer")
plt.ylabel("Catching Ball Reward")
plt.ylim(-0.5,2.5)

plt.legend()
plt.savefig("./plots/units-u-shape-catch.png")

# for decoding attention window

aw_performance_vis_input = np.array([0.6005,0.56,0.573])
aw_performance_vis_input_schema_pos = np.array([0.85,0.79,0.7763]
)

plt.figure()
plt.bar(x-25, aw_performance_vis_input_schema_pos, width=50, label="Visual Input and AS Position")
plt.bar(x+25, aw_performance_vis_input, width=50, color="r", label="Only Visual Input")
plt.xticks(x)
plt.xlabel("Units per Layer")
plt.ylabel("Decoding Accuracy")
plt.ylim(0,1.1)
plt.legend()
plt.savefig("./plots/units-u-shape-decoding_aw.png")

aw_decoding_differences = aw_performance_vis_input_schema_pos - aw_performance_vis_input
catching_differences = np.array(means_catch) - np.array(means_catch_random_schema)
att_differences = np.array(means_att) - np.array(means_att_random_schema)

plt.figure()
plt.scatter(aw_decoding_differences, att_differences, s=50, label="Attention Reward")
plt.scatter(aw_decoding_differences, catching_differences, s=50, color="r", label="Catching Reward")
#plt.xticks(x)
plt.xlabel("Differences in Decoding AW Position Accuracy", fontsize = 12)
plt.ylabel("Differences in Rewards", fontsize = 12)
#plt.ylim(0,1.1)
plt.legend()


plt.savefig("./plots/units-u-shape-differences.png")