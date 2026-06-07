Wybrany przez nas temat projektu to XAI for Network Intrusion Detection.
 
Artykułem bazowym jest:

XAI-IDS: Toward Proposing an Explainable Artificial Intelligence Framework for Enhancing Network Intrusion Detection Systems (Arreche et al., 2024)

DOI: https://doi.org/10.3390/app14104170
 
W ramach projektu planujemy implementację:

- Modelu Random Forest do klasyfikacji ruchu sieciowego na podstawie artykułu

- Techniki XAI - SHAP (Shapley Additive Explanations) do generowania wyjaśnień

- Dataset NSL-KDD
 
Planowane porównania dotyczyć będą:

- Wyników modelu (accuracy, F1-score) z oryginalnym artykułem

- Najważniejszych cech identyfikowanych przez SHAP
 
Jakie są oczekiwane wnioski: 

SHAP powinien wskazać podobne kluczowe cechy co w oryginale (np. bajty na pakiet, czas połączenia).
 
W ramach eksperymentów planujemy:

1. Porównanie SHAP między Random Forest a XGBoost - czy oba modele identyfikują te same najważniejsze cechy?

2. Test na datasetcie UNSW-NB15 - czy wyjaśnienia są spójne między datasetami?
 
Dodatkowa Uwaga: Równolegle uczestniczymy w zajęciach z SIWY(wyjaśnialna sztuczna inteligencja). Chętnie połączylibyśmy tematy o ile okaże się to możliwe. Dokumentacja projektu na SIWY: https://github.com/FilipBudzynski/SIWY/blob/main/design_proposal.md. Temati XAI bardzo nam odpowiada ze względu na równoległe uczestnictwo na powyższym przedmiocie.
 
Z wyrazamia szacunku,

Dawid Budzyński, Filip Budzyński

Zespół 13
XAI-IDS: Toward Proposing an Explainable Artificial Intelligence Framework for Enhancing Network Intrusion Detection Systems
The exponential growth of network intrusions necessitates the development of advanced artificial intelligence (AI) techniques for intrusion detection systems (IDSs). However, the reliance on AI for...
 
dziękuję za przesłane opracowanie - ogólnie jest ok, więc konsultacje mają Panowie zaliczone
 
ale proszę wybrać 2 różne datasety, ale nie KDD ani NSL-KDD, bo te mają ponad 25 lat 
 
ten UNSW-NB15 jest ok, a do tego trzeba dobrać też jeszcze inny (nowszy)
 
ten KDD może być jedynie po to żeby porównać się z wynikami z artykułu
