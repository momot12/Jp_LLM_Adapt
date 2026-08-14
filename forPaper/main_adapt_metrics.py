from adapt_metrics import (store_formality_per_model_forPaper,
                           filter_genders_forPaper,
                           calc_f1_acc_formality_filtered_forPaper)


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
    
    
    
    for i in range(0, len(DIRS), 5):
        # slicing model groups to only calculate chi2 among same models
        current_group = DIRS[i : i + 5]
        print(current_group)
        
        combined_basic_data = store_formality_per_model_forPaper(dirs=current_group)
        filtered_data, sp_gender_raw = filter_genders_forPaper(per_model_dict=combined_basic_data,
                                                system_prompt_gender="neutral",
                                                question_gender="neutral")
        calc_f1_acc_formality_filtered_forPaper(filtered_output=filtered_data)
