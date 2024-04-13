# ML Fagkveld Praktisk


I denne praktiske oppgaven ønsker vi å trene en modell som klassifiserer hvilken klasse ulike nyhetsoverskrifter hører til. Vi blir kun gitt `topic` og `title` i `dataset.csv`, noe som kompliserer problemet. For å kunne trene på titlene er man nødt til å få disse over på et format en modell kan trenes på (tall). 

Det oppfordres til å benytte Python, men det er lagd et spor for eventuelle .NET entusiaster. Python kan lastes ned her: https://www.python.org/downloads/. 

Hint: 
- Dette er et flerklasse klassifiseringsproblem. Les mer om det [her](https://scikit-learn.org/stable/modules/multiclass.html).
- W3 schools kan hjelpe deg med pandas: https://www.w3schools.com/python/pandas/default.asp
- [ML.NET dokumentason](https://learn.microsoft.com/en-gb/dotnet/machine-learning/)


## Steg for å løse oppgaven:
1. Last inn datasett
2. Undersøk datasett
3. Generer embeddings
4. Tren modell
5. Valider modell
6. (Ekstra for Python spor) Visualisering
7. (Ekstra for Python spor) Clustering



Man kan følge to spor:
1. Python i `supervised_learning.ipynb`, følg oppgavene nedover.
    - `AZURE_OPENAI_API_KEY` og `AZURE_OPENAI_ENDPOINT` blir lagt ut på Slack.
2. .NET åpne prosjektet i MLFagkveld og følg oppskriften i `Program.cs`


### ML.NET instruksjoner:
Installer ml.net for ditt os:
```bash
# OSX x64
dotnet tool install -g mlnet-osx-x64

# OSX arm
dotnet tool install -g mlnet-osx-arm64

# Windows: https://dotnet.microsoft.com/en-us/learn/ml-dotnet/get-started-tutorial/install
```