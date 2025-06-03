# Gerekli kütüphanelerin import edilmesi
import pandas as pd
import numpy as np
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, matthews_corrcoef
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms

# Pandas gelecek sürüm uyarılarını engelleme
try:
    pd.set_option('future.no_silent_downcasting', True)
except:
    pass  # Eski pandas sürümlerinde bu seçenek mevcut değil

# DataFrame görüntüleme ayarları
pd.set_option('display.max_columns', None)  # Tüm sütunları göster
pd.set_option('display.max_rows', None)     # Tüm satırları göster

# Adım 1: Veri Setinin Yüklenmesi ve Ön İşleme

# Veri setini yükle
df = pd.read_csv('New_1000_Sales_Records.csv')
print("Veri setinin ilk 5 satırı:")
print(df.head())
print("\nVeri seti hakkında genel bilgiler:")
print(df.info())
print("\nVeri seti boyutu:")
print(df.shape)
print("\nSayısal değişkenlerin istatistiksel özeti:")
print(df.describe())

# İkili hedef değişkeni oluştur: High_Profit (Total Profit medyanın üstündeyse 1, değilse 0)
df['High_Profit'] = (df['Total Profit'] > df['Total Profit'].median()).astype(int)
print("\nHedef değişkenin sınıf dağılımı:")
print(df['High_Profit'].value_counts())

# Sipariş-Gönderim günlerini sayısal formata çevir (örn: "5 days" -> 5)
df['Order_Ship_Days'] = df['Order_Ship_Days'].str.extract(r'(\d+)').astype(int)

# Özellik seçimi (Ülke hariç)
features_all = ['Region', 'Item Type', 'Sales Channel', 'Order Priority',
                'Units Sold', 'Unit Price', 'Unit Cost', 'Total Revenue', 'Total Cost',
                'Total Profit', 'Order year', 'Order Month', 'Order Weekday', 'Unit Margin',
                'Order_Ship_Days']

# PCA için kullanılacak özellikler (Total Revenue, Cost ve Profit hariç)
features_pca = ['Region', 'Item Type', 'Sales Channel', 'Order Priority',
                'Units Sold', 'Unit Price', 'Unit Cost', 'Order year', 'Order Month',
                'Order Weekday', 'Unit Margin', 'Order_Ship_Days']

# Özellik ve hedef matrislerini oluştur
X = df[features_all]
X_pca = df[features_pca]
y = df['High_Profit']

# One-Hot Encoding işlemi
# Kategorik değişkenleri kukla değişkenlere dönüştür (ilk kategoriyi düşür)
X = pd.get_dummies(X, columns=['Region', 'Item Type', 'Sales Channel', 'Order Priority', 'Order Weekday'], drop_first=True)
X_pca = pd.get_dummies(X_pca, columns=['Region', 'Item Type', 'Sales Channel', 'Order Priority', 'Order Weekday'], drop_first=True)

# Sayısal değişkenleri ölçeklendir
numerical_cols = ['Units Sold', 'Unit Price', 'Unit Cost', 'Total Revenue', 'Total Cost',
                  'Total Profit', 'Order year', 'Order Month', 'Unit Margin', 'Order_Ship_Days']
numerical_cols_pca = [col for col in numerical_cols if col not in ['Total Revenue', 'Total Cost', 'Total Profit']]

# PCA öncesi korelasyon kontrolü
print("\nTotal Profit ile özelliklerin korelasyonu:")
print(X[['Total Revenue', 'Total Cost', 'Total Profit']].corr())

# Ön işleme sonrası özellikleri yazdır
print("\nTüm özellikler (ön işleme sonrası):")
print("X özellikleri:")
print(X.columns.tolist())
print("\nX şekli:", X.shape)

print("\nPCA için özellikler:")
print(X_pca.columns.tolist())
print("\nX_pca şekli:", X_pca.shape)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Sayısal değişkenleri ölçeklendir
numerical_cols = ['Units Sold', 'Unit Price', 'Unit Cost', 'Total Revenue', 'Total Cost',
                'Total Profit', 'Order year', 'Order Month', 'Unit Margin', 'Order_Ship_Days']
numerical_cols_pca = [col for col in numerical_cols if col not in ['Total Revenue', 'Total Cost', 'Total Profit']]

# Ana veri seti için ölçeklendirme
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# PCA veri seti için ölçeklendirme
scaler_pca = StandardScaler()
X_train_pca[numerical_cols_pca] = scaler_pca.fit_transform(X_train_pca[numerical_cols_pca])
X_test_pca[numerical_cols_pca] = scaler_pca.transform(X_test_pca[numerical_cols_pca])

# Adım 2: Özellik Mühendisliği ve Model Hazırlığı

# Genetik Algoritma ile Özellik Seçimi için değerlendirme fonksiyonu
def evaluate_features(individual, X, y):
    # Eşik değeri 0.5 olarak belirlendi - bu değerin üzerindeki özellikler seçilir
    mask = np.array([1 if gene > 0.5 else 0 for gene in individual], dtype=bool)

    # Hiç özellik seçilmediyse en kötü skor döndür
    if not any(mask):
        return -1.0,

    # Seçilen özellikleri al
    X_selected = X.iloc[:, mask]

    # Çok fazla özellik seçilirse ceza uygula
    feature_penalty = -0.01 * len(X_selected.columns)  # Her fazla özellik için -0.01 ceza

    # k-NN sınıflandırıcı ile değerlendirme
    clf = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(clf, X_selected, y, cv=5, scoring='f1')

    return scores.mean() + feature_penalty,  # Skor ve ceza toplamını döndür

# GA için genetik algoritma kurulumu
if 'FitnessMax' not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if 'Individual' not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

# GA araç kutusunun hazırlanması
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(X.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_features, X=X_train, y=y_train)
toolbox.register("mate", tools.cxTwoPoint)  # İki noktalı çaprazlama operatörü
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # Yerleşik mutasyon operatörü
toolbox.register("select", tools.selTournament, tournsize=3)  # Turnuva seçimi

# Genetik algoritmayı çalıştır
population = toolbox.population(n=50)  # 50 bireylik popülasyon
NGEN = 20  # 20 nesil boyunca evrim
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

# En iyi bireyi seç
best_individual = tools.selBest(population, k=1)[0]
selected_features_ga = X.columns[np.array(best_individual, dtype=bool)].tolist()
print("Genetik Algoritma ile seçilen özellikler:", selected_features_ga)

# Seçilen özelliklerle veri setlerini oluştur
X_train_ga = X_train[selected_features_ga]
X_test_ga = X_test[selected_features_ga]

# Özgüllük (Specificity) metriğini hesaplama fonksiyonu
def specificity_score(y_true, y_pred):
    """
    Özgüllük (Specificity) metriğini hesaplar.

    Parametreler:
    y_true: Gerçek etiketler
    y_pred: Tahmin edilen etiketler

    Dönüş:
    float: Özgüllük skoru (TN / (TN + FP))
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# k-NN için hiperparametre optimizasyonu
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
grid_search_knn = GridSearchCV(knn, param_grid, cv=10, scoring='f1')
grid_search_knn.fit(X_train, y_train)
print(f"En iyi k değeri: {grid_search_knn.best_params_}")

# Sınıflandırıcıları tanımla
classifiers = {
    'k-NN': grid_search_knn.best_estimator_,
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(max_depth=2, min_samples_split=50, min_samples_leaf=20, random_state=42),
    'Linear SVM': SVC(kernel='linear', random_state=42),
    'RBF SVM': SVC(kernel='rbf', random_state=42),
    'Poly SVM': SVC(kernel='poly', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'Random Forest': RandomForestClassifier(max_depth=2, min_samples_split=50, min_samples_leaf=20, n_estimators=20, max_features='sqrt', random_state=42)
}

# Adım 3: Model Değerlendirme ve Tablolar

# Tablo 1: Standart Model Performansı
results = []
for name, clf in classifiers.items():
    # Modeli eğit ve test et
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    scores = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Specificity': specificity_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }
    results.append(scores)

table1 = pd.DataFrame(results)[['Model', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'MCC', 'F1 Score']]
print("Tablo 1: Standartlaştırılmış Sınıflandırıcı Algoritmaların Performansı")
print(table1)
table1.to_csv('table1_classifier_performance.csv', index=False)

# Tablo 2: Genetik Algoritma ile Özellik Seçimi
ga_results = []
for name, clf in classifiers.items():
    clf.fit(X_train_ga, y_train)
    y_pred_ga = clf.predict(X_test_ga)

    scores = {
        'Model': name,
        'Seçilen Özellik Sayısı': len(selected_features_ga),
        'Accuracy': accuracy_score(y_test, y_pred_ga),
        'Precision': precision_score(y_test, y_pred_ga),
        'Recall': recall_score(y_test, y_pred_ga),
        'Specificity': specificity_score(y_test, y_pred_ga),
        'MCC': matthews_corrcoef(y_test, y_pred_ga),
        'F1 Score': f1_score(y_test, y_pred_ga)
    }
    ga_results.append(scores)

table2 = pd.DataFrame(ga_results)
print("\nTablo 2: Genetik Algoritma ile Özellik Seçimi Performans Sonuçları")
print(table2)
table2.to_csv('table2_ga_performance.csv', index=False)

# Tablo 3: PCA ile Özellik Azaltma
# PCA uygula - Önceden ayrılmış olan eğitim ve test setleri üzerinde
pca = PCA(n_components=0.90)
X_train_pca_transformed = pca.fit_transform(X_train_pca)  # Sadece eğitim seti üzerinde fit_transform
X_test_pca_transformed = pca.transform(X_test_pca)  # Test seti üzerinde sadece transform

# PCA sonuçlarını görselleştir
plt.figure(figsize=(12, 6))
plt.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_)
plt.xlabel('Bileşen Numarası')
plt.ylabel('Açıklanan Varyans Oranı')
plt.title('Her Bir PCA Bileşeninin Açıkladığı Varyans Oranı')
plt.xticks(range(1, pca.n_components_ + 1))
plt.grid(True)
plt.savefig('pca_variance_ratios.png')
plt.close()

# Kümülatif varyans grafiği
plt.figure(figsize=(12, 6))
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, pca.n_components_ + 1), cumsum, 'bo-')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('Kümülatif Açıklanan Varyans Oranı')
plt.title('PCA Bileşenleri - Kümülatif Varyans')
plt.grid(True)
plt.xticks(range(1, pca.n_components_ + 1))
plt.savefig('pca_cumulative_variance.png')
plt.close()

# PCA performans sonuçlarını hesapla
pca_results = []
for name, clf in classifiers.items():
    clf.fit(X_train_pca_transformed, y_train)
    y_pred_pca = clf.predict(X_test_pca_transformed)

    scores = {
        'Model': name,
        'PCA Bileşen Sayısı': X_train_pca_transformed.shape[1],
        'PCA Açıklanan Varyans': pca.explained_variance_ratio_.sum(),
        'Accuracy': accuracy_score(y_test, y_pred_pca),
        'Precision': precision_score(y_test, y_pred_pca),
        'Recall': recall_score(y_test, y_pred_pca),
        'Specificity': specificity_score(y_test, y_pred_pca),
        'MCC': matthews_corrcoef(y_test, y_pred_pca),
        'F1 Score': f1_score(y_test, y_pred_pca)
    }
    pca_results.append(scores)

# PCA sonuçlarını yazdır ve kaydet
print("\nTablo 3: PCA ile Özellik Azaltma Performans Sonuçları")
table3 = pd.DataFrame(pca_results)
print(table3)
table3.to_csv('table3_pca_performance.csv', index=False)

# Adım 4: Detaylı Model Analizi

# Seçili modeller için karışıklık matrisi ve sınıflandırma raporu
for name, clf in [('k-NN', classifiers['k-NN']), ('Naive Bayes', classifiers['Naive Bayes'])]:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n{name} Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

    # Karışıklık matrisini görselleştir
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Karışıklık Matrisi')
    plt.savefig(f'{name}_confusion_matrix.png')
    plt.close()

# Karar ağacı görselleştirmeleri
dt = classifiers['Decision Tree']
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, class_names=['Düşük Kar', 'Yüksek Kar'], filled=True)
plt.savefig('decision_tree_visualization.png')
plt.close()

# Total Profit olmadan karar ağacı analizi
X_no_profit = X.drop('Total Profit', axis=1)
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_no_profit, y, test_size=0.2, random_state=42)
dt_np = DecisionTreeClassifier(max_depth=2, min_samples_split=50, min_samples_leaf=20, random_state=42)
dt_np.fit(X_train_np, y_train_np)
y_pred_np = dt_np.predict(X_test_np)
print("\nTotal Profit olmadan Karar Ağacı F1 Skoru:", f1_score(y_test_np, y_pred_np))

plt.figure(figsize=(20, 10))
plot_tree(dt_np, feature_names=X_no_profit.columns, class_names=['Düşük Kar', 'Yüksek Kar'], filled=True)
plt.title('Karar Ağacı Görselleştirmesi (Total Profit olmadan)')
plt.savefig('decision_tree_no_profit_visualization.png')
plt.close()