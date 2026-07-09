from wilcoxon_signed_rank import calc_wilcoxon_signed_rank
from format_for_wilcoxon import stack_2700_points, get_900_points_per_t, get_formalities_perfile



if __name__ == "__main__":
    DIRS = [
        "jp_stable_1_6B",
        "jp_stable_1_6B_female",
        "jp_stable_1_6B_male",
        "jp_stable_1_6B_nanako",
        "jp_stable_1_6B_rintaro",
        "jp_stable_7B",
        "jp_stable_7B_female",
        "jp_stable_7B_male",
        "jp_stable_7B_nanako",
        "jp_stable_7B_rintaro",
        "qwen_1_7B",
        "qwen_1_7B_female",
        "qwen_1_7B_male",
        "qwen_1_7B_nanako",
        "qwen_1_7B_rintaro",
        "qwen_4B",
        "qwen_4B_female",
        "qwen_4B_male",
        "qwen_4B_nanako",
        "qwen_4B_rintaro"
    ]
    
    threshold_bleu = 0.8
    
    for i in range(0, len(DIRS), 5):
        # slicing model groups to only calculate chi2 among same models
        current_group = DIRS[i : i + 5]
        print(current_group)

        in_out_dict_baseline = get_formalities_perfile(sub_dir=current_group[0], threshold_bleu=threshold_bleu)
        in_out_dict_female = get_formalities_perfile(sub_dir=current_group[1], threshold_bleu=threshold_bleu)
        in_out_dict_male = get_formalities_perfile(sub_dir=current_group[2], threshold_bleu=threshold_bleu)
        in_out_dict_nanako = get_formalities_perfile(sub_dir=current_group[3], threshold_bleu=threshold_bleu)
        in_out_dict_rintaro = get_formalities_perfile(sub_dir=current_group[4], threshold_bleu=threshold_bleu)

        # get dict with {t=1: [900 points], t=2: [900 points], t=3:[900 points]}
        baseline_900_dict = get_900_points_per_t(data=in_out_dict_baseline)
        female_900_dict = get_900_points_per_t(data=in_out_dict_female)
        male_900_dict = get_900_points_per_t(data=in_out_dict_male)
        nanako_900_dict = get_900_points_per_t(data=in_out_dict_nanako)
        rintaro_900_dict = get_900_points_per_t(data=in_out_dict_rintaro)

        # get stacked for one number verdict => 900,900,900 = 2700
        baseline_2700_array = stack_2700_points(points_dict=baseline_900_dict)
        female_2700_array = stack_2700_points(points_dict=female_900_dict)
        male_2700_array = stack_2700_points(points_dict=male_900_dict)
        nanako_2700_array = stack_2700_points(points_dict=nanako_900_dict)
        rintaro_2700_array = stack_2700_points(points_dict=rintaro_900_dict)


        # global (2700): calc Wilcoxon for each pair (total 4)
        calc_wilcoxon_signed_rank(baseline_scores=baseline_2700_array, prompt_scores=female_2700_array, prompt_name="female")
        calc_wilcoxon_signed_rank(baseline_scores=baseline_2700_array, prompt_scores=male_2700_array, prompt_name="male")
        calc_wilcoxon_signed_rank(baseline_scores=baseline_2700_array, prompt_scores=nanako_2700_array, prompt_name="nanako")
        calc_wilcoxon_signed_rank(baseline_scores=baseline_2700_array, prompt_scores=rintaro_2700_array, prompt_name="rintaro")



