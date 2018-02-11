# architecture-style-recognition

# Tim

Ana Rudić, SW 47/2014<br>
Milan Šalić, SW 53/2014<br>
Sreten Stokić, SW 44/2014

# Definicija problema

Prepoznavanje arhitektonskog stila određene građevine.

# Motivacija problema

Prepoznati stil nekon arhitektonskog objekta često je lak zadatak za arhitekte, umjetnike i uopšte poštovaoce kulture. Ipak, postoje i ostali koji nisu toliko upoznati sa arhitekturom, a htjeli bi znati kom stilu pripada neka, njima zanimljiva, građevina.

# Skup podataka

Skup podataka moguće je prikupiti na internetu, koji je prepun slika građevina raznih arhitektonskih stilova. Za svaki stil potrebno je pronaći određeni broj slika građevina, a zatim od svake slike napraviti nekoliko sličnih, koje prikazuju građevinu sa te slike u drugačijim veličinama i sa različitim zakrivljenostima.<br>
<br>
Google drive sa materijalima: <a>https://drive.google.com/drive/u/1/folders/1bgVhjeXlksvBG3NPCzDyY_oWH3jg3W5a</a>

# Metodologija

Potrebno je iskoristiti jedan od algoritama za izdvajanje objekata od pozadine kako bi se građevina dovela u fokus, a za to bi bio pogodan thresholding. Da bi se dobro uočile ivice građevine, planirano je da se koristi Canny operator. Takođe potrebna je i neuronska mreža za obučavanje za prepoznavanje stilova.

# Metod evaluacije

Mjeriće se tačnost (Accuracy) prepoznavanja.
