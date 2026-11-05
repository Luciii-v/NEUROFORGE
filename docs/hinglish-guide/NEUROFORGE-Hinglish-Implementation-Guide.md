# NEUROFORGE — The Builders' Playbook

**Build. Mutate. Ship.** · 4 November 2026

This guide explains the current models, Architect workflow, CNN concepts, metrics and implementation roadmap in Hinglish.

## 01. NEUROFORGE ka current model setup

Abhi hum exactly kya use kar rahe hain?

### Short answer

NEUROFORGE mein main model ek PyTorch DynamicCNN hai. Architect iske architecture ko mutate karta hai. ResNet-18 ek separate transfer-learning comparison baseline hai.

| Model | Role | Current status |
| --- | --- | --- |
| DynamicCNN | Mutation search ka selected winner | 74.19% latest saved proxy accuracy; 2 proxy epochs |
| Manual CNN | Fixed human-designed reference | 50.00% accuracy; 2-epoch proxy |
| ResNet-18 | ImageNet-pretrained comparison baseline | 70.79% accuracy; 20 final epochs |

Decision tree, Random Forest, XGBoost, SVM, KNN ya Logistic Regression abhi project mein use nahi ho rahe. Raw images ke pixels par tree models usually best first choice nahi hote; unko CNN embeddings ke upar classical baseline ki tarah add kiya ja sakta hai.

### Important

ResNet mutation engine ka part nahi hai. Woh ek fixed backbone comparison hai. Isse hum poochte hain: evolved small CNN aur standard transfer-learning model ke results kaise compare hote hain?

## 02. Audio classification ka complete flow

Sound ko tensor mein kaise badalte hain, aur model kaise predict karta hai?

### Short answer

Audio mein hum waveform ko directly classifier ko nahi dete. Pehle WAV ko mono 16 kHz par laate hain, 4-second clip fit karte hain, phir log-mel spectrogram banate hain. Yeh one-channel image jaisa tensor DynamicCNN ko diya jata hai.

| Stage | Simple Hinglish meaning |
| --- | --- |
| WAV input | 16-bit PCM WAV clip; pehle audio file read hoti hai. |
| Mono + resample | Multiple channels average hote hain aur sample rate 16,000 Hz par aata hai. |
| Crop / pad | Clip ko 4 seconds ke fixed length mein center-crop ya zero-pad karte hain. |
| STFT | Short windows par frequency analysis karke time ke saath frequency changes milte hain. |
| Mel filterbank | Frequencies ko 64 hearing-inspired mel bands mein group karte hain. |
| Log + normalize | Energy range compress aur per-clip scale normalize hoti hai. |
| DynamicCNN | 1 × 64 × roughly 251 tensor se sound class logits predict hote hain. |

### Viva line

Image mein pixels ka pattern hota hai; audio mein time-frequency pattern. Dono ko CNN ke liye tensor banakar alag classifiers train karte hain.

## 03. Audio architecture aur benchmark plan

Architecture same idea follow karti hai, lekin input channel alag hota hai.

| Architecture part | Image path | Audio path |
| --- | --- | --- |
| Input | 3-channel RGB image | 1-channel log-mel spectrogram |
| Feature blocks | Conv2D → BatchNorm/activation → optional pooling/dropout | Same DynamicCNN blocks over frequency-time features |
| Compression | Adaptive average pooling | Adaptive average pooling |
| Head | Linear layer with image classes | Linear layer with sound classes |
| Output | Softmax class probabilities | Softmax class probabilities |

1. Pehle ESC-10 subset par plumbing smoke test karo.
2. ESC-50 metadata aur official folds verify karo.
3. Folds 1–3 train, fold 4 validation, fold 5 first test check ke liye use karo.
4. Small mutation search run karke tensor shape, labels aur checkpoint reload verify karo.
5. Real benchmark ke liye official cross-validation aggregation add karo.
6. Audio weights, config, ordered labels, preprocessing aur fold metadata separate export karo.

### Current status

Audio pipeline ka core foundation code mein hai. React audio upload/playback, FastAPI audio prediction route aur real ESC-50 benchmark abhi next work hain. Synthetic WAV check benchmark accuracy prove nahi karta.

## 04. Architect ka workflow

Architect ka kaam: blueprint choose karna, weights learn karna nahi.

1. Dataset mount hota hai: local path ya ZIP se labelled images read hote hain.
2. Classes aur split metadata inspect hota hai.
3. Manual CNN baseline run hota hai, taaki fixed reference mile.
4. Generation 1 mein default blueprint aur uske mutated children bante hain.
5. Har candidate ko short proxy budget par train kiya jata hai.
6. Accuracy, precision, recall, F1, parameter count aur latency record hote hain.
7. Fitness score candidates ko rank karta hai.
8. Top-half parents select hote hain; best design ka elite copy preserve hota hai.
9. Children parent architecture se mutate hote hain; trained weights inherit nahi hote.
10. Duplicate architecture cache se repeated evaluation avoid hoti hai.
11. Final winner ko longer full-training budget par fresh weights ke saath train kiya jata hai.
12. DynamicCNN, manual CNN aur ResNet-18 ko same active dataset/split context mein compare kiya jata hai.

### Ek line mein

Outer loop blueprint evolve karta hai. Inner loop us blueprint ke weights learn karta hai.

## 05. CNN layer-by-layer

Convolutional layer kaun hai, aur har layer ka kya kaam hai?

| Component | Simple Hinglish meaning |
| --- | --- |
| Input | Image ko numbers/tensor mein convert kiya jata hai; current image path 64×64 preprocessing use karta hai. |
| Convolutional layer | Image par chhote windows slide karke local patterns detect karta hai: edges, texture, shape. |
| Filter | Ek learned detector. Har filter kisi pattern ko detect karne ki koshish karta hai. |
| Kernel | Filter ka small spatial window, jaise 3×3 ya 5×5. Kernel size batata hai detector kitna area dekhta hai. |
| Activation | Non-linearity add karta hai. ReLU, GELU aur SiLU current search choices hain. |
| BatchNorm | Intermediate activations ko stable scale par rakhta hai, training ko easier bana sakta hai. |
| Pooling | Feature map ka spatial size reduce karta hai. MaxPool efficiency badhata hai, detail lose kar sakta hai. |
| Dropout2d | Training ke time kuch feature channels temporarily remove karta hai, overfitting reduce karne ke liye. |
| Adaptive average pooling | Har feature channel ko ek summary number mein convert karta hai, image size variations handle karne mein help karta hai. |
| Linear classifier | Final layer har class ke liye ek score/logit deta hai. |
| Softmax | Logits ko relative class probabilities mein convert karta hai; confidence correctness guarantee nahi hai. |

## 06. Filters, kernels, dropout aur normalization

In words ko viva mein kaise explain karein?

### Filter vs kernel

Kernel window ka size hai: 3×3 ya 5×5. Filter poora learned detector hai. Ek convolutional layer mein multiple filters hote hain; har filter ek output feature map banata hai.

### Dropout

Dropout model ko ek hi shortcut par depend karne se rokne ki koshish karta hai. Yeh sirf training time par active hota hai; inference mein disabled hota hai.

### Normalization

Input normalization image values ko consistent numerical range mein laata hai. Batch normalization network ke beech ke activations ko stabilize karta hai. Dono same cheez nahi hain.

| Example | Meaning |
| --- | --- |
| filters [32, 64] | Pehli convolutional layer ke 32 detectors, doosri ke 64 detectors. |
| kernel_sizes [3, 5] | Pehli layer 3×3 window aur doosri 5×5 window use karegi. |
| dropout 0.4 | Training mein feature channels ka roughly 40% dropout probability. |
| batch_norm true | Convolution ke baad BatchNorm enabled. |
| pooling false | Architecture mein MaxPool branches disabled. |

## 07. Classification vs regression

Hum kya kar rahe hain, aur MSE/RMSE ka actual meaning kya hai?

| Topic | Classification | Regression |
| --- | --- | --- |
| Output | Known classes mein se ek label, jaise Apple ya Banana | Ek continuous number, jaise price ya temperature |
| Current NEUROFORGE use | Yes: multiclass image classification | No: current production task regression nahi hai |
| Typical loss | Cross-entropy | MSE or Huber loss |
| Typical metrics | Accuracy, precision, recall, F1 | MSE, RMSE, MAE, R² |

Current comparison table mein MSE/RMSE classification probability vectors par calculate hote hain: predicted probability vector ko one-hot true-label vector se compare kiya jata hai. Yeh auxiliary diagnostics hain, regression proof nahi. MSE/RMSE lower hona better hai.

### Viva answer

Hum classification solve kar rahe hain. Accuracy aur macro F1 main metrics hain. MSE/RMSE ko probability calibration/error diagnostics ke roop mein show karte hain, regression metrics ke roop mein nahi.

## 08. Current results aur >90% roadmap

Target possible ho sakta hai, guarantee nahi.

| Model | Acc | Precision | Recall | F1 | MSE | RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| DynamicCNN / 2 proxy | 74.19% | 71.07% | 69.95% | 67.47% | 0.0417 | 0.2043 |
| Manual CNN / 2 proxy | 50.00% | 52.61% | 46.10% | 43.22% | 0.0797 | 0.2824 |
| ResNet-18 / 20 final | 70.79% | 70.25% | 62.01% | 64.01% | 0.0479 | 0.2189 |

1. 64×64 se 128×128 ya 224×224 resolution compare karo; small visual details preserve honge.
2. ResNet-18 ka final block unfreeze karke low learning rate par fine-tune karo.
3. EfficientNet, ResNet-50 ya ConvNeXt jaise stronger pretrained baselines add karo.
4. Architect ko learning rate, weight decay, resolution aur augmentation bhi mutate/tune karne do.
5. Labels, duplicates, imbalance aur source leakage audit karo.
6. At least 3 seeds par repeat karo; same split, budget aur hardware report karo.
7. Ensemble tabhi use karo jab validation aur untouched test par improvement prove ho.

### Important math

Latest DynamicCNN row 2-epoch proxy result hai, final independent test nahi. 89-image split par har ek prediction percentage ko noticeably move kar sakti hai. 90% ko target samjho, guarantee nahi; pehle independent split aur repeated seeds complete karo.

## 09. Kya Decision Tree ya Random Forest add karna chahiye?

Classical ML ka sensible comparison kaise banega?

Raw image pixels directly tree models ko dena usually weak baseline hoga: dimensionality bahut high hoti hai aur spatial structure lose ho jata hai.

1. Pretrained CNN se har image ka embedding vector extract karo.
2. Embedding ko train/validation/test split ke saath store karo.
3. Random Forest, SVM, Logistic Regression aur XGBoost ko same embedding par train karo.
4. Same class labels, same test split aur same metrics use karo.
5. Result ko DynamicCNN/ResNet se alag classical baseline ke roop mein label karo.

### Current status

Decision Tree, Random Forest aur SVM abhi implement nahi hain. Inko add karna possible hai, lekin pehle current CNN/ResNet comparison ko stable aur fair rakhna better hai.

## 10. Implementation timeline

11 baje phone kholkar jo idea aaya tha, uska status.

### Clarification note

“11 baje phone kholkar aaya hua idea” ka exact feature name repository notes mein recorded nahi hai. Isliye us feature ke baare mein fake technical claim nahi karenge.

| Phase | Kab / priority | What happens |
| --- | --- | --- |
| Current release | Abhi complete | Metric comparison, ResNet-18 baseline, MSE/RMSE, checkpoint downloads and stale-dataset guard. |
| Next experiment | Next implementation sprint | 224px resolution, ResNet fine-tuning, learning-rate/augmentation tuning and repeated seeds. |
| Phone-open idea | Next sprint after exact feature confirmation | One-line acceptance criteria define karke prototype, test and demo integration. |
| Classical baselines | After CNN comparison is stable | CNN embeddings par Random Forest/SVM/Logistic Regression comparison. |

### Short answer

Pehle current comparison ko fair baseline banayenge. Uske baad >90% improvement experiments. Phone-open idea ka exact naam/expected behavior confirm hote hi usko next sprint mein implement karenge.

## 11. Viva cheat sheet

Short answers jo confidently bol sakte ho.

| Question | Answer |
| --- | --- |
| What is NEUROFORGE? | Mutation-based CNN architecture-search studio with visible lineage, training evidence and local inference. |
| Architect kya karta hai? | Candidate CNN blueprints generate, mutate, rank and select karta hai. |
| Weights inherit hote hain? | Nahi. Child architecture inherit karta hai; weights fresh train hote hain. |
| ResNet-18 kyun? | Fixed pretrained transfer-learning baseline, mutation winner prove karne ke liye nahi. |
| Random Forest use ho raha hai? | Nahi. Current system DynamicCNN, manual CNN aur ResNet-18 use karta hai. |
| Classification ya regression? | Multiclass image classification. MSE/RMSE probability-vector diagnostics hain. |
| >90% kaise? | Better data, higher resolution, fine-tuning, stronger baselines, hyperparameter search and repeated fair evaluation. |

### Remember

Accuracy ka number akela enough nahi. Dataset, model, split, training budget aur all metrics saath mein bolne hain.

