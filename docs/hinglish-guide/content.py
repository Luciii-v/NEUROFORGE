"""Editorial Hinglish content for the NEUROFORGE field playbook."""

PAGES = []


def page(title, subtitle, blocks, tag="FIELD NOTES", source=""):
    PAGES.append({"title": title, "subtitle": subtitle, "blocks": blocks, "tag": tag, "source": source})


def note(title, body):
    return ("note", title, body)


def qa(title, body):
    return ("qa", title, body)


def callout(title, body):
    return ("callout", title, body)


page("NEUROFORGE\nTHE BUILDERS' PLAYBOOK", "Architecture, image + audio classification, metrics aur viva roadmap — simple language mein.", [("cover",)], tag="FIELD NOTES / 2026")

page("Abhi hum kya use kar rahe hain?", "Current model stack ko ek glance mein samjho.", [
    callout("SHORT ANSWER", "Main model PyTorch <b>DynamicCNN</b> hai. Architect iske architecture ko mutate karta hai. <b>ResNet-18</b> separate transfer-learning comparison baseline hai."),
    ("table", ["Model", "Role", "Current evidence"], [
        ["DynamicCNN", "Mutation-search winner", "74.19% latest saved proxy accuracy; 2 proxy epochs"],
        ["Manual CNN", "Fixed human reference", "50.00% accuracy; 2-epoch proxy"],
        ["ResNet-18", "Pretrained comparison baseline", "70.79% accuracy; 20 final epochs"],
    ]),
    note("Classical ML abhi nahi hai", "Decision Tree, Random Forest, XGBoost, SVM, KNN aur Logistic Regression current production workflow mein use nahi ho rahe. Inko baad mein CNN embeddings ke upar fair baseline ki tarah add kar sakte hain."),
    callout("IMPORTANT", "ResNet mutation engine ka part nahi hai. Woh fixed backbone comparison hai: evolved CNN aur standard transfer learning ko same dataset/split par compare karne ke liye."),
], tag="MODEL STACK", source="server.py /api/comparison; neuroforge_core.py; frontend/src/components/MutationLoopPage.jsx.")

page("Audio classification ka flow", "Sound ko log-mel tensor mein convert karke DynamicCNN ko diya jata hai.", [
    ("table", ["Stage", "Simple Hinglish meaning"], [
        ["WAV input", "16-bit PCM WAV clip read hoti hai."],
        ["Mono + 16 kHz", "Channels average aur sample rate normalize hota hai."],
        ["4-second fit", "Long clip center-crop, short clip zero-pad hoti hai."],
        ["STFT + mel", "Time-frequency energy ko 64 mel bands mein map karte hain."],
        ["Log + normalize", "Energy compress karke feature scale stable banate hain."],
        ["CNN output", "1 × 64 × roughly 251 feature map se sound class logits milte hain."],
    ]),
    callout("VIVA LINE", "Image mein spatial pixels hote hain; audio mein time-frequency patterns. Input adapter alag hai, lekin architecture search ka outer loop same hai."),
], tag="AUDIO FOUNDATION", source="audio_pipeline.py: WAV decoder, STFT, mel filterbank and AudioClassificationDataset.")

page("Audio architecture aur next work", "Core code available hai; product workflow aur benchmark complete karna baaki hai.", [
    ("compare", [
        ("IMAGE", "Input: 3-channel RGB.<br/>DynamicCNN feature blocks.<br/>Adaptive average pooling.<br/>Image-class linear head."),
        ("AUDIO", "Input: 1-channel log-mel map.<br/>Same DynamicCNN block choices.<br/>Adaptive average pooling.<br/>Sound-class linear head."),
    ]),
    ("steps", [
        ("01", "ESC-10 subset par smoke test."),
        ("02", "ESC-50 official folds aur WAV compatibility verify."),
        ("03", "Audio FastAPI prediction route aur React upload/playback/spectrogram view add."),
        ("04", "Audio search, final training aur separate artifact bundle verify."),
        ("05", "Official fold aggregation aur repeated seeds ke baad benchmark claim."),
    ]),
    callout("CURRENT BOUNDARY", "Synthetic WAV check plumbing prove karta hai, accuracy nahi. Real ESC-50/UrbanSound8K benchmark ke bina audio quality claim nahi karenge."),
], tag="AUDIO ROADMAP", source="audio_pipeline.py; server.py; frontend/src; docs/dataset-and-training-plan.md.")

page("Architect ka workflow", "Architect ka kaam blueprint choose karna hai — weights learn karna nahi.", [
    ("flow", [
        ("01  MOUNT DATA", "Local path ya ZIP se labelled images read hote hain."),
        ("02  INSPECT", "Classes, sample count aur dataset format check hota hai."),
        ("03  BASELINE", "Fixed Manual CNN reference ke liye evaluate hota hai."),
        ("04  BUILD", "Default CNN blueprint aur mutated children generate hote hain."),
        ("05  PROXY TRAIN", "Har candidate short budget par fresh weights ke saath train hota hai."),
        ("06  SCORE", "Accuracy, precision, recall, F1, parameters aur latency record hote hain."),
        ("07  SELECT", "Fitness ke basis par top parents aur elite preserve hote hain."),
        ("08  MUTATE", "Children parent architecture se nayi traits try karte hain."),
        ("09  FINAL TRAIN", "Winner ko declared full-training epochs par fresh weights milte hain."),
        ("10  COMPARE + SHIP", "DynamicCNN, Manual CNN aur ResNet-18 side-by-side report hote hain."),
    ]),
    callout("EK LINE MEIN", "Outer loop blueprint evolve karta hai. Inner loop us blueprint ke weights learn karta hai. Child architecture inherit karta hai, trained weights nahi."),
], tag="ARCHITECT / THE TOURNAMENT", source="architect.py: mutate_config, run_architect_search; server.py; MutationLoopPage.jsx.")

page("CNN layer-by-layer", "Convolutional layer kaun hai, aur har part ka kya kaam hai?", [
    note("Assembly line", "<b>Input → Convolution → optional BatchNorm → Activation → optional MaxPool → optional Dropout2d → Adaptive Average Pool → Linear Classifier</b>"),
    ("table", ["Component", "Simple Hinglish meaning"], [
        ["Input", "Image numbers/tensor mein convert hoti hai; current image preprocessing 64×64 use karta hai."],
        ["Convolutional layer", "Small windows slide karke edges, texture aur shapes jaise local patterns detect karta hai."],
        ["Filter", "Learned detector; har filter kisi useful pattern ko detect karne ki koshish karta hai."],
        ["Kernel", "Filter ka spatial window — 3×3 ya 5×5 — jo batata hai kitna local area dekha ja raha hai."],
        ["Activation", "Non-linearity add karta hai; ReLU, GELU aur SiLU search choices hain."],
        ["BatchNorm", "Intermediate activations ko stable scale par rakhta hai."],
        ["Pooling", "Feature map chhota karta hai; efficiency badhti hai, detail lose ho sakti hai."],
        ["Dropout2d", "Training mein feature channels temporarily remove karke overfitting reduce karta hai."],
        ["Linear classifier", "Har class ke liye final score/logit deta hai."],
        ["Softmax", "Scores ko relative probabilities mein convert karta hai; confidence correctness guarantee nahi hai."],
    ]),
], tag="MODEL IDENTITY", source="neuroforge_core.py: DynamicCNN, _build_model.")

page("Filters, kernels, dropout aur normalization", "Viva mein in terms ko kaise explain karna hai?", [
    callout("FILTER VS KERNEL", "Kernel window ka size hai: 3×3 ya 5×5. Filter poora learned detector hai. Ek layer mein multiple filters hote hain, aur har filter ek feature map banata hai."),
    callout("DROPOUT", "Dropout model ko sirf ek shortcut par depend karne se rokne ki koshish karta hai. Training time par active hota hai; inference time par disabled hota hai."),
    callout("NORMALIZATION", "Input normalization image values ko consistent numerical range mein laata hai. Batch normalization network ke beech ke activations stabilize karta hai. Dono same cheez nahi hain."),
    ("table", ["Config example", "Meaning"], [
        ["filters [32, 64]", "Layer 1 mein 32 detectors, layer 2 mein 64 detectors."],
        ["kernel_sizes [3, 5]", "Layer 1 3×3 aur layer 2 5×5 window use karegi."],
        ["dropout 0.4", "Training mein feature-channel dropout probability 40%."],
        ["batch_norm true", "Convolution ke baad BatchNorm enabled."],
        ["pooling false", "MaxPool branches disabled."],
    ]),
], tag="CNN VOCABULARY", source="neuroforge_core.py DynamicCNN; best_model_config.json.")

page("Classification vs regression", "Hum kya solve kar rahe hain, aur MSE/RMSE ka role kya hai?", [
    ("compare", [
        ("CLASSIFICATION", "Known classes mein se label choose karna.<br/><b>Current use:</b> fruit/pest image classification.<br/><b>Typical metrics:</b> accuracy, precision, recall, F1.<br/><b>Loss:</b> cross-entropy."),
        ("REGRESSION", "Continuous number predict karna, jaise price ya temperature.<br/><b>Current use:</b> nahi.<br/><b>Typical metrics:</b> MSE, RMSE, MAE, R²."),
    ]),
    note("MSE/RMSE current table mein", "Hum MSE/RMSE ko predicted probability vector aur one-hot true label ke beech calculate karte hain. Yeh probability-error diagnostics hain, regression proof nahi. MSE/RMSE lower hona better hota hai."),
    callout("VIVA ANSWER", "Hum multiclass classification solve kar rahe hain. Accuracy aur macro F1 main metrics hain. MSE/RMSE ko probability-vector diagnostics ke roop mein show karte hain."),
], tag="METRICS / MEANING", source="neuroforge_core.py: classification_metrics; server.py /api/comparison.")

page("Current results aur >90% roadmap", "Target possible ho sakta hai, lekin guarantee nahi.", [
    ("table", ["Model", "Acc", "Precision", "Recall", "F1", "MSE", "RMSE"], [
        ["DynamicCNN / 2 proxy", "74.19%", "71.07%", "69.95%", "67.47%", "0.0417", "0.2043"],
        ["Manual CNN / 2 proxy", "50.00%", "52.61%", "46.10%", "43.22%", "0.0797", "0.2824"],
        ["ResNet-18 / 20 final", "70.79%", "70.25%", "62.01%", "64.01%", "0.0479", "0.2189"],
    ]),
    ("steps", [
        ("01", "64×64 ke saath 128×128 ya 224×224 compare karo; small details preserve honge."),
        ("02", "ResNet-18 ka final block unfreeze karke low learning rate par fine-tune karo."),
        ("03", "EfficientNet, ResNet-50 ya ConvNeXt jaise stronger pretrained baselines add karo."),
        ("04", "Architect ko learning rate, weight decay, resolution aur augmentation tune karne do."),
        ("05", "Labels, duplicates, imbalance aur source leakage audit karo."),
        ("06", "At least 3 seeds par repeat karo; same split, budget aur hardware report karo."),
    ]),
    callout("SMALL TEST SET MATH", "Latest DynamicCNN row 2-epoch proxy result hai, final independent test nahi. 89-image split par har prediction percentage ko noticeably move kar sakti hai; 90% ko target samjho, guarantee nahi."),
], tag="RESULTS / ROADMAP", source="best_model_metadata.json; resnet18_comparison_metadata.json; /api/comparison.")

page("Decision Tree ya Random Forest?", "Classical ML ko sensible comparison ki tarah kaise add karenge?", [
    note("Current status", "Decision Tree, Random Forest, XGBoost, SVM, KNN aur Logistic Regression abhi implement nahi hain. Current production stack DynamicCNN, Manual CNN aur ResNet-18 hai."),
    ("steps", [
        ("01", "Pretrained CNN se har image ka embedding vector extract karo."),
        ("02", "Embedding ko train/validation/test split ke saath store karo."),
        ("03", "Random Forest, SVM, Logistic Regression aur XGBoost ko same embeddings par train karo."),
        ("04", "Same labels, same split aur same six metrics use karo."),
        ("05", "Result ko CNN/ResNet se alag classical baseline label karo."),
    ]),
    callout("WHY NOT RAW PIXELS?", "Raw image pixels high-dimensional hote hain aur spatial structure lose ho jata hai. CNN embeddings par classical models zyada sensible comparison banenge."),
], tag="CLASSICAL BASELINES", source="Current implementation scope; future experiment design.")

page("11 baje phone kholkar jo idea aaya tha", "Implementation timeline — bina fake claim ke.", [
    callout("CLARIFICATION NOTE", "“11 baje phone kholkar aaya hua idea” ka exact feature name repository notes mein recorded nahi hai. Isliye us feature ke baare mein invented technical details nahi likh rahe."),
    ("table", ["Phase", "Priority", "What happens"], [
        ["Current release", "Complete", "Metric comparison, ResNet-18, MSE/RMSE, downloads aur stale-dataset guard."],
        ["Next experiment", "Next sprint", "224px resolution, ResNet fine-tuning, learning-rate/augmentation tuning aur repeated seeds."],
        ["Phone-open idea", "After exact confirmation", "Acceptance criteria define karke prototype, test aur demo integration."],
        ["Classical baselines", "After comparison stabilizes", "CNN embeddings par Random Forest/SVM/Logistic Regression."],
    ]),
    callout("SHORT ANSWER", "Pehle current comparison ko fair baseline banayenge. Phir >90% improvement experiments. Phone idea ka exact naam aur expected behavior confirm hote hi next sprint mein implement hoga."),
], tag="IMPLEMENTATION PLAN", source="Project state and pending feature clarification.")

page("Viva cheat sheet", "Short answers jo confidently bol sakte ho.", [
    ("table", ["Question", "Answer"], [
        ["What is NEUROFORGE?", "Mutation-based CNN architecture-search studio with visible lineage, training evidence and local inference."],
        ["Architect kya karta hai?", "Candidate CNN blueprints generate, mutate, rank aur select karta hai."],
        ["Weights inherit hote hain?", "Nahi. Child architecture inherit karta hai; weights fresh train hote hain."],
        ["ResNet-18 kyun?", "Fixed pretrained transfer-learning baseline; mutation winner prove karne ke liye nahi."],
        ["Random Forest use ho raha hai?", "Nahi. Future embedding-based classical baseline ho sakta hai."],
        ["Classification ya regression?", "Multiclass image classification. MSE/RMSE probability-vector diagnostics hain."],
        [">90% kaise?", "Better data, higher resolution, fine-tuning, stronger baselines, hyperparameter search aur repeated fair evaluation."],
    ]),
    callout("REMEMBER", "Accuracy ka number akela enough nahi. Dataset, model, split, training budget aur all metrics saath mein bolne hain."),
], tag="LAST-MINUTE MODE", source="NEUROFORGE current workflow and comparison artifacts.")
