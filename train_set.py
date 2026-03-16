train_library = [
    "book/MobyDick_sub_grafi/mobydick_kge.tsv",
    "book/JekyllHyde_sub_grafi/JekyllHyde.tsv",
    "book/Burmese_Days/Burmese_Days.tsv",
    "book/TheVerdict_sub_grafi/theverdict_kge.tsv",
    "book/A_Room_Of_Ones_own/A_Room_Of_Ones_Own.tsv",
    "book/Dracula/Dracula.tsv",
    "book/The_Time_Machine/The_Time_Machine.tsv",
    "book/1984/1984.tsv", #  ultimo aggiunto 
    "book/TheGreatGatsby/TheGreatGatsby.tsv", #  ultimo aggiunto 
    "book/MadameBovary/MadameBovary.tsv", #  ultimo aggiunto 
]

train_params_1 = {
    "hidden_channels": 32,
    "num_layers": 3,
    "dropout_rate": 0.10174214482282506,
    "lr": 0.008846175564081828,
    "weight_decay": 2.6756844783872094e-05,
    "margin": 0.5,
    "k_negatives": 3,
    "val_loss" : 0.1445
}

train_params_2 = {
    "hidden_channels": 64,
    "num_layers": 2,
    "dropout_rate": 0.22943336121970184,
    "lr": 0.0018238113989311065,
    "weight_decay": 0.0001164486298898722,
    "margin": 0.5,
    "k_negatives": 4
}

train_params_3 = {
    "val_loss": 0.1303,
    "hidden_channels": 128,
    "num_layers": 3,
    "dropout_rate": 0.4553512742575813,
    "lr": 0.0046605004599490595,
    "weight_decay": 0.0043494798614568736,
    "margin": 0.5,
    "k_negatives": 2
}

test_library = [
    "book/Animal_Farm/Animal_Farm.tsv",
    "book/I_Am_Legend/I_Am_Legend.tsv",
    "book/A_Vision_Of_Judgment/A_Vision_Of_Judgment.tsv",
    "book/The_Empire_Of_The_Ants/The_Empire_Of_The_Ants.tsv",
    "book/TenderIsTheNight/TenderIsTheNight.tsv" # ultimo aggiunto 
]