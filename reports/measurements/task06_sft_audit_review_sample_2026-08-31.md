# Przegląd oznaczeń audytu SFT — po 10 losowych na oś (2026-08-31)

Próbka do ręcznej weryfikacji trafności etykiet sędziego (Qwen3.8-27B).
Losowanie seedem 20260831 z pełnych zbiorów wadliwych; osie nie wykluczają
się nawzajem, więc para może wystąpić w dwóch sekcjach. Przy każdej pozycji
werdykt sędziego dla wszystkich czterech osi.

## Nieodpowiadalne — pasaż nie zawiera odpowiedzi na zapytanie  (w próbce: 1833/12000)

### nieodpowiadalne #1  (`pair_id=54026::960531`)

**Zapytanie:** co to jest stopień wzrostu ciśnienia w hamulcach pneumatycznych

**Pasaż:** Sprawdź szybkość narastania ciśnienia powietrza: Gdy silnik pracuje na obrotach na minutę, ciśnienie powinno wzrosnąć od 85 do 100 psi w ciągu 45 sekund w systemach z podwójnym powietrzem. (Jeśli pojazd ma zbiorniki powietrza większe niż minimalne, czas nagrzewania może być dłuższy i nadal bezpieczny.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=nie; główny problem: `nieodpowiadalne`

### nieodpowiadalne #2  (`pair_id=232578::3626240`)

**Zapytanie:** jaki jest kod pocztowy dla hamilton nj

**Pasaż:** Kod pocztowy 08619 znajduje się w stanie New Jersey w obszarze metra w Filadelfii. Kod pocztowy 08619 znajduje się głównie w hrabstwie Mercer. Oficjalna nazwa US Postal Service dla 08619 to TRENTON, New Jersey. Fragmenty kodu pocztowego 08619 znajdują się w granicach miasta Mercerville-Hamilton Square, NJ, Trenton, NJ, lub graniczą z nimi. Numer kierunkowy dla kodu pocztowego 08619 to 609.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=nie; główny problem: `nieodpowiadalne`

### nieodpowiadalne #3  (`pair_id=366553::1433923`)

**Zapytanie:** ile strzelanin na kampusach uniwersyteckich w 2015

**Pasaż:** Po dwóch strzelaninach na kampusach uniwersyteckich w piątek liczba strzelanin w szkołach w USA w 2015 r. wzrosła do 52, z 30 osób zabitych i 53 innych rannych. Te sumy obejmują samobójstwa, incydenty, w których nikt nie został trafiony, oraz jeden atak na autobus szkolny.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=nie; główny problem: `nieodpowiadalne`

### nieodpowiadalne #4  (`pair_id=446039::6451474`)

**Zapytanie:** jak długo trwa urlop ojcowski w tennessee

**Pasaż:** Zalety Tennessee FLA. Prawo TN skupia się wyłącznie na ciąży, urlopie macierzyńskim i adopcji i dopuszcza 16 tygodni urlopu w porównaniu z 12 tygodniami w przypadku prawa federalnego. Jeśli pojawią się potrzeby niezwiązane z ciążą, urlopem macierzyńskim i adopcją, te dwa przepisy mogą obowiązywać kolejno, a nie jednocześnie.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=nie; główny problem: `nieodpowiadalne`

### nieodpowiadalne #5  (`pair_id=380::62686`)

**Zapytanie:** przyczyny odmy podskórnej

**Pasaż:** Uraz klatki piersiowej, główna przyczyna rozedmy podskórnej, może powodować przedostawanie się powietrza do skóry klatki piersiowej z szyi lub płuc. W przypadku nakłucia błon opłucnowych, jak to ma miejsce przy urazie penetrującym klatki piersiowej, powietrze może przedostawać się z płuc do mięśni i tkanki podskórnej ściany klatki piersiowej.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=nie; główny problem: `nieodpowiadalne`

### nieodpowiadalne #6  (`pair_id=392816::790800`)

**Zapytanie:** jak obliczyć odsetki składane w sposób ciągły

**Pasaż:** Definicja ciągłego łączenia. Ciągłe składanie ma miejsce, gdy odsetki są naliczane od kapitału i składane w sposób ciągły, to znaczy, że odsetki są stale dodawane do kapitału, aby ponownie naliczane były odsetki.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=nie; główny problem: `nieodpowiadalne`

### nieodpowiadalne #7  (`pair_id=437211::6321674`)

**Zapytanie:** jak wykorzystuje się istotne dane

**Pasaż:** z odpowiednimi danymi, które będą wykorzystywane przez praktyka pracownika i można uzyskać do nich dostęp za pomocą przenośnego terminala lub HHT. Kiedy farmaceuta realizuje receptę, upewnia się, że lek ma etykietę z kodem kreskowym, zanim opuści jego aptekę.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=tak; główny problem: `zbyt_ogolne`

### nieodpowiadalne #8  (`pair_id=285267::4378013`)

**Zapytanie:** kim byli Merowingowie

**Pasaż:** Przyczyna i skutek… — Merowing. Merowing (czasami nazywany Francuzem) to stary, potężny program, który znajduje się w Matrixie. Opisywany jako handlarz informacji, Merowing zachowuje się jak przywódca potężnego syndykatu przestępczości zorganizowanej.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=nie; główny problem: `nieodpowiadalne`

### nieodpowiadalne #9  (`pair_id=147298::2374946`)

**Zapytanie:** jak długo powinienem jechać trasą ogrodów

**Pasaż:** 1...Parkuj i jedź. Mógłbyś przejechać Garden Route w jeden dzień, ale ja nie. Zatrzymaj się w miejscu takim jak Mossel Bay, z najłagodniejszym klimatem na świecie przez cały rok po Hawajach lub Plettenberg Bay, ze swoimi wspaniałymi, złotymi plażami. Lub Knysna, uznane za najlepsze małe miasteczko w RPA, ze względu na galerie sztuki, kawiarnie i restauracje… Parkuj i jedź. Mógłbyś przejechać Garden Route w jeden dzień, ale ja nie. Zatrzymaj się w miejscu takim jak Mossel Bay, o najłagodniejszym na świecie całorocznym klimacie po Hawajach.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=nie; główny problem: `nieodpowiadalne`

### nieodpowiadalne #10  (`pair_id=444563::6429213`)

**Zapytanie:** Holiday Inn Wabash w

**Pasaż:** Położony w północno-wschodniej części stanu Indiana, Holiday Inn Express Hotel Wabash jest dogodnie zlokalizowany w pobliżu Manchester College, Honeywell Center, zbiorników Salamonie i Mississinewa oraz Huntington College.

*Werdykt:* odpowiadalne=nie, polszczyzna=nie, sensowne=nie, zbyt_ogólne=nie; główny problem: `tlumaczenie`

## Zła polszczyzna — kalki, mojibake, przekręcone nazwy po tłumaczeniu  (w próbce: 1388/12000)

### zla_polszczyzna #1  (`pair_id=79972::1983847`)

**Zapytanie:** czym jest super g ski

**Pasaż:** Super G jest skrótem od Super Giant Slalom i jest imprezą narciarstwa alpejskiego, łączącą narciarstwo zjazdowe z nawigacją po trasach slalomowych. Ten sport po raz pierwszy stał się popularny w latach 70., a swoją premierę miał jako zawody Pucharu Świata w 1982 roku. Wymaga umiejętności zarówno w narciarstwie zjazdowym, jak i slalomie, i jest prawdopodobnie jednym z najbardziej intensywnych zawodów w narciarstwie alpejskim. Ważna jest duża szybkość i dokładność w zakrętach, gdyż pominięcie zakrętu wokół bramki złożonej z dwóch tyczek od razu dyskwalifikuje zawodników. Istnieją pewne różnice m…

*Werdykt:* odpowiadalne=tak, polszczyzna=nie, sensowne=tak, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### zla_polszczyzna #2  (`pair_id=393681::5893505`)

**Zapytanie:** w jakim hrabstwie znajduje się McComb w stanie Missisipi

**Pasaż:** McComb i otaczający go obszar hrabstwa Pike mają trzy oddzielne okręgi szkolne, trzy szkoły prywatne i kolegium społeczne w północnej części hrabstwa.

*Werdykt:* odpowiadalne=tak, polszczyzna=nie, sensowne=tak, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### zla_polszczyzna #3  (`pair_id=79972::4142200`)

**Zapytanie:** czym jest super g ski

**Pasaż:** Super slalom gigant lub super-G to dyscyplina wyścigowa narciarstwa alpejskiego. Wraz z szybszym zjazdem jest uważany za zawody szybkościowe, w przeciwieństwie do zawodów technicznych, slalomu giganta i slalomu.

*Werdykt:* odpowiadalne=tak, polszczyzna=nie, sensowne=tak, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### zla_polszczyzna #4  (`pair_id=277541::1628727`)

**Zapytanie:** jaka jest populacja pismo beach ca

**Pasaż:** Plaża Pismo w Kalifornii. Pismo Beach to miasto w hrabstwie San Luis Obispo, w środkowej części wybrzeża Kalifornii, w Stanach Zjednoczonych. Szacunkowa populacja wynosiła 7931 w 2014 r., w porównaniu z 7655 w spisie z 2010 r. Jest częścią Obszaru Pięciu Miast, skupiska miast w tym obszarze hrabstwa San Luis Obispo.

*Werdykt:* odpowiadalne=tak, polszczyzna=nie, sensowne=tak, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### zla_polszczyzna #5  (`pair_id=362355::5458695`)

**Zapytanie:** jak zrobić ach stop

**Pasaż:** Na przykład niektóre banki umożliwiają klientom dokonywanie postojów ACH telefonicznie lub osobiście, podczas gdy inne akceptują przefaksowany formularz zatrzymania płatności. Aby przetworzyć zatrzymanie, klient podaje informacje o koncie, nazwę sprzedawcy i dokładną kwotę płatności. Opłata za zatrzymanie czeków różni się w zależności od instytucji.

*Werdykt:* odpowiadalne=tak, polszczyzna=nie, sensowne=tak, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### zla_polszczyzna #6  (`pair_id=17046::378617`)

**Zapytanie:** pogoda w newport de

**Pasaż:** Newport Prognoza pogody â 10 dni. 1 Newport 1–3 dniowa prognoza pogody Podsumowanie: Głównie sucho. Ciepło (max 24,9°C we wtorek po południu, min 4,2°C w czwartek wieczorem). 2 Newport Prognoza pogody na 4–7 dni Podsumowanie: Lekki deszcz (łącznie 2 mm), głównie w niedzielny poranek. Ciepło (max 24,9°C w niedzielne popołudnie, min 5,9°C w piątek rano).

*Werdykt:* odpowiadalne=tak, polszczyzna=nie, sensowne=tak, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### zla_polszczyzna #7  (`pair_id=268935::4146364`)

**Zapytanie:** co to jest selenium browser

**Pasaż:** Selenium to przenośna platforma testowania oprogramowania dla aplikacji internetowych. Selenium to zestaw różnych narzędzi programowych, z których każde ma inne podejście do wspierania automatyzacji testów. Selenium jest biblioteką lub reprezentuje zestaw interfejsów API, których można użyć do programowego sterowania przeglądarką.

*Werdykt:* odpowiadalne=tak, polszczyzna=nie, sensowne=tak, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### zla_polszczyzna #8  (`pair_id=224769::3514769`)

**Zapytanie:** gdzie kręcony jest walking dead

**Pasaż:** Gdzie jest kręcony The Walking Dead? Dom Ricka Grimesa, zakład karny West Georgia Correctional Facility i dom Hershela zostały nakręcone w pobliżu Atlanta Georgia

*Werdykt:* odpowiadalne=tak, polszczyzna=nie, sensowne=tak, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### zla_polszczyzna #9  (`pair_id=383809::5782901`)

**Zapytanie:** czy tętnice pępkowe są utlenowane

**Pasaż:** Łożysko połączone jest z płodem pępowiną. Wewnątrz sznurka znajdują się dwie tętnice i żyła; obie tętnice są określane jako tętnice pępowinowe. Dwie tętnice otaczają żyłę pępowinową, która przenosi natlenioną krew i składniki odżywcze z łożyska do płodu. Po wykorzystaniu tlenu i składników odżywczych tętnica pępowinowa przenosi odtlenioną krew i odpady z płodu z powrotem do łożyska.

*Werdykt:* odpowiadalne=tak, polszczyzna=nie, sensowne=tak, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### zla_polszczyzna #10  (`pair_id=153646::2467849`)

**Zapytanie:** zdefiniuj termin off the grid

**Pasaż:** poza siecią. 1 W lub w sytuacji lub miejscu, w którym energia elektryczna z publicznego systemu elektroenergetycznego nie jest wykorzystywana. ( 2 idiomatic) W lub w tajnej lub odizolowanej sytuacji lub miejscu, zwłaszcza w takim, w którym komunikacja publiczna jest ograniczona. (3 idiomatyczne) Potajemnie; potajemnie.

*Werdykt:* odpowiadalne=tak, polszczyzna=nie, sensowne=tak, zbyt_ogólne=nie; główny problem: `tlumaczenie`

## Zbyt ogólne — zapytanie pasuje do tysięcy pasaży  (w próbce: 282/12000)

### zbyt_ogolne #1  (`pair_id=138137::2238803`)

**Zapytanie:** czy DNA jest widoczne

**Pasaż:** Białka DNA i histonów są upakowane w struktury zwane chromosomami. W jądrze każdej komórki cząsteczka DNA jest upakowana w nitkowate struktury zwane chromosomami. Każdy chromosom składa się z DNA ciasno owiniętego wielokrotnie wokół białek zwanych histonami, które wspierają jego strukturę. Chromosomy nie są widoczne w komórce jądro – nawet pod mikroskopem – gdy komórka się nie dzieli. Jednak DNA tworzące chromosomy staje się ciaśniej upakowane podczas podziału komórki i jest wtedy widoczne pod mikroskopem. W jądrze każdej komórki, cząsteczce DNA jest pakowany w nitkowate struktury zwane chromo…

*Werdykt:* odpowiadalne=tak, polszczyzna=tak, sensowne=tak, zbyt_ogólne=tak; główny problem: `zbyt_ogolne`

### zbyt_ogolne #2  (`pair_id=443647::6416234`)

**Zapytanie:** podaj definicję

**Pasaż:** Dostarczac; dostarczać; zapewnić do użytku. Delp przeciwko Brewing Co., 123 Pa. 42, 15Atl. 871; Wyatt przeciwko Larimer i W. Irr. Co., 1 kolor. 480. 29 os. 906. Użyte w przepisach dotyczących alkoholu: dostarczanie środków na zaopatrzenie w jakikolwiek sposób i obejmuje dawanie i sprzedaż. .Stan przeciwko Freemanowi, 27 Vt. 520; Stan przeciwko Tague, 76 Vt 118, 56 Atl.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=nie, zbyt_ogólne=tak; główny problem: `zbyt_ogolne`

### zbyt_ogolne #3  (`pair_id=186347::2951732`)

**Zapytanie:** jak leczy się raka pęcherza moczowego

**Pasaż:** Leczenie raka pęcherza w stadium IV (przerzutowym) będzie na ogół składać się z chemioterapii, badań klinicznych lub połączenia tych dwóch. Można przeprowadzić operację w celu usunięcia dotkniętych węzłów chłonnych lub złagodzenia późnych objawów choroby.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=tak; główny problem: `zbyt_ogolne`

### zbyt_ogolne #4  (`pair_id=81352::1380649`)

**Zapytanie:** jak używać paracordu

**Pasaż:** Lekki i mocny paracord może być używany do projektów rzemieślniczych, takich jak smycze i bransoletki. Jeśli chcesz połączyć dwa kawałki paracordu, możesz zrobić atrakcyjne połączenie z płomieniem. Stopione złącze będzie mocniejsze i bardziej atrakcyjne niż węzeł. Lekkie i mocne, paracord może być używany do projektów rzemieślniczych, takich jak smycze i bransoletki. Jeśli chcesz połączyć dwa kawałki paracordu, możesz zrobić atrakcyjne połączenie z płomieniem. Skondensowane połączenie będzie mocniejsze i bardziej atrakcyjne niż węzeł.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=tak; główny problem: `zbyt_ogolne`

### zbyt_ogolne #5  (`pair_id=437211::6321674`)

**Zapytanie:** jak wykorzystuje się istotne dane

**Pasaż:** z odpowiednimi danymi, które będą wykorzystywane przez praktyka pracownika i można uzyskać do nich dostęp za pomocą przenośnego terminala lub HHT. Kiedy farmaceuta realizuje receptę, upewnia się, że lek ma etykietę z kodem kreskowym, zanim opuści jego aptekę.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=tak; główny problem: `zbyt_ogolne`

### zbyt_ogolne #6  (`pair_id=261741::4045205`)

**Zapytanie:** z czego jest zrobiony drut

**Pasaż:** Uważam, że zwykle druty są wykonane z miedzi. Czy większość przewodów jest wykonana z miedzi? Czy rozmiar drutu zwykle określa, jaki metal znajduje się w drucie? Na przykład, czy większe druty są zwykle wykonane z miedzi, podczas gdy mniejsze druty są zwykle wykonane z żelaza? Czy żelazo jest kiedykolwiek używane w drutach? Czy stal jest kiedykolwiek używana w drutach? Czy w drutach stosuje się aluminium? Chciałbym uzyskać ogólny przegląd rodzajów metali w przewodach.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=tak; główny problem: `zbyt_ogolne`

### zbyt_ogolne #7  (`pair_id=174076::2770236`)

**Zapytanie:** zdefiniuj mikro

**Pasaż:** Mikro: Mini, mały, malutki. Micro jest również terminem używanym w RTS Starcraft, jako termin na zmodyfikowaną kontrolę jednostki. Mikro może również odnosić się do opisów mikroorganizmów. Mikro jako jednostka miary jest zdefiniowana jako: Jedna milionowa...

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=tak; główny problem: `zbyt_ogolne`

### zbyt_ogolne #8  (`pair_id=53743::955896`)

**Zapytanie:** czy jest to forma niedbalstwa zawodowego

**Pasaż:** W angielskim prawie dotyczącym czynów niedozwolonych zaniedbanie zawodowe stanowi podzbiór ogólnych zasad dotyczących zaniedbania, obejmujący sytuację, w której pozwany reprezentował siebie jako posiadający ponadprzeciętne umiejętności i zdolności.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=tak; główny problem: `zbyt_ogolne`

### zbyt_ogolne #9  (`pair_id=203828::3208736`)

**Zapytanie:** dlaczego mamy trzecie strony

**Pasaż:** Ale mniejsze partie lub osoby trzecie często odgrywały rolę w polityce. Strony trzecie skupiają uwagę na problemach i pomysłach. Czasami uzyskują wystarczające poparcie, aby wpłynąć na wynik wyborów. Czasami strona trzecia osiąga część swoich celów, wspierając główną stronę, która obiecuje działać zgodnie z poglądami strony trzeciej. Po wojnie domowej Amerykanie dyskutowali o kwestiach takich jak prawa wyborcze kobiet i reforma pracy. Nowe partie polityczne pomogły skupić uwagę na tych kwestiach.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=tak; główny problem: `zbyt_ogolne`

### zbyt_ogolne #10  (`pair_id=88365::1489069`)

**Zapytanie:** jakie napoje można zrobić z rumem korzennym

**Pasaż:** 3/4 uncji rumu Captain Morgan® Oryginalny przyprawiony rum. 1/2 uncji likieru migdałowego amaretto. 1/2 uncji soku żurawinowego. 1/4 uncji słodko-kwaśnej mieszanki. Do shakera wypełnionego do połowy kostkami lodu wlej amaretto, rum z przyprawami, sok żurawinowy i kwaśną mieszankę. Dobrze wstrząśnij i przelej do staromodnej szklanki wypełnionej w 1/4 kostkami lodu.

*Werdykt:* odpowiadalne=nie, polszczyzna=tak, sensowne=tak, zbyt_ogólne=tak; główny problem: `zbyt_ogolne`

## Niesensowne — nie wygląda na realną potrzebę informacyjną  (w próbce: 84/12000)

### niesensowne #1  (`pair_id=290514::4452412`)

**Zapytanie:** co oznacza rozwój telencephalonu od

**Pasaż:** W rozwoju prenatalnym przodomózgowie (promózgowie) jest najdalej wysuniętym do przodu pęcherzykiem cewy nerwowej, który później tworzy zarówno międzymózgowia, jak i kresomózgowia (który rozwija się w mózg). Składa się z kilku struktur między pniem mózgu a mózgiem, w tym wzgórza , podwzgórze, nabłonek, przysadka mózgowa i szyszynka. Embriologicznie, międzymózgowiem zaczyna się jako obszar cewy nerwowej embrionu kręgowca, w którym powstają struktury tylnego przodomózgowia.

*Werdykt:* odpowiadalne=nie, polszczyzna=nie, sensowne=nie, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### niesensowne #2  (`pair_id=194200::3067777`)

**Zapytanie:** hrabstwo Iowa to Unionville

**Pasaż:** Unionville to miasto w hrabstwie Appanoose w stanie Iowa w Stanach Zjednoczonych. Według spisu z 2010 roku populacja wynosiła 102.

*Werdykt:* odpowiadalne=nie, polszczyzna=nie, sensowne=nie, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### niesensowne #3  (`pair_id=32449::622192`)

**Zapytanie:** co oznacza rozejm z byłym

**Pasaż:** Przegląd haseł słownikowych: Co oznacza rozejm? â ROZEJM (rzeczownik) Rzeczownik ROZEJM ma 1 znaczenie: 1. stan pokoju uzgodniony między przeciwnikami, aby mogli dyskutować o warunkach pokojowych Informacje ogólne: ROZEJM używany jako rzeczownik jest bardzo rzadki.

*Werdykt:* odpowiadalne=nie, polszczyzna=nie, sensowne=nie, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### niesensowne #4  (`pair_id=318145::4839797`)

**Zapytanie:** amylopectin jest formą skrobi, która ma ________

**Pasaż:** Obecność amylopektyny można określić za pomocą testu jodowego. Amylopektyna jest rozpuszczalna w wodzie, ma duże zdolności wiązania i uczestniczy w retrogradacji skrobi. Retrogradacja skrobi to proces przemiany skrobi z ciekłego roztworu w żel.

*Werdykt:* odpowiadalne=nie, polszczyzna=nie, sensowne=nie, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### niesensowne #5  (`pair_id=249942::3874968`)

**Zapytanie:** co było zbudowane z wieży eiffla

**Pasaż:** Wieża Eiffla, po francusku La Tour Eiffel, była głównym eksponatem Wystawy Paryskiej – czyli Wystawy Światowej – w 1889 roku. Została zbudowana, aby upamiętnić stulecie Rewolucji Francuskiej i zademonstrować światu przemysłową potęgę Francji .

*Werdykt:* odpowiadalne=nie, polszczyzna=nie, sensowne=nie, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### niesensowne #6  (`pair_id=95913::1604230`)

**Zapytanie:** czy żyrafa urodziła się w przygodzie

**Pasaż:** 28 października 2014 o 10:10, aktualizacja 28 października 2014 o 10:21. Drukuj e-mail. Uwagi. JACKSON – Mała żyrafa o imieniu Mika, mierząca metr osiemdziesiąt i dziesięć cali, urodziła się w Six Flags Great Adventure na początku tego miesiąca, stając się dziewiątą żyrafą w parku Safari Off Road Adventure, poinformowali we wtorek przedstawiciele parku.

*Werdykt:* odpowiadalne=tak, polszczyzna=nie, sensowne=nie, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### niesensowne #7  (`pair_id=473824::1705167`)

**Zapytanie:** pięciu gości z five guys

**Pasaż:** Siedziba główna firmy Five Guys w Lorton w stanie Wirginia. Five Guys zostało założone w 1986 roku przez Janie i Jerry Murrell; Jerry i synowie pary, Jim, Matt, Chad i Ben, byli oryginalnymi Five Guys. Murrellowie mieli piątego syna, Tylera, dwa lata później.

*Werdykt:* odpowiadalne=nie, polszczyzna=nie, sensowne=nie, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### niesensowne #8  (`pair_id=99284::1656043`)

**Zapytanie:** co to jest koszykowy przypadek

**Pasaż:** Basket Case to piosenka amerykańskiego zespołu punk rockowego Green Day. Jest to siódmy utwór i trzeci singiel z trzeciego studyjnego albumu Dookie (1994). Piosenka spędziła pięć tygodni na szczycie listy Billboard's Alternative Songs.

*Werdykt:* odpowiadalne=nie, polszczyzna=nie, sensowne=nie, zbyt_ogólne=nie; główny problem: `tlumaczenie`

### niesensowne #9  (`pair_id=159302::2551491`)

**Zapytanie:** czy zęby krokodyla są znajdowane w?

**Pasaż:** Duży kredowy ząb kopalny krokodyla - Maroko. Jest to duży, wiekowy, skamieniały ząb krokodyla z okresu kredy, pochodzący z łóżek Kem Kem w Maroku. Wielu sprzedawców etykietuje podobne zęby jako pochodzące od sarkozucha (The Super Croc), aby ułatwić ich sprzedaż. W basenie Kem Kem żyło kilka gatunków krokodyli.

*Werdykt:* odpowiadalne=nie, polszczyzna=nie, sensowne=nie, zbyt_ogólne=tak; główny problem: `tlumaczenie`

### niesensowne #10  (`pair_id=404992::6002329`)

**Zapytanie:** ebbe wat is

**Pasaż:** Ponadto Ebbe to zwierzę domowe o imieniu Asbjoern (AsbjÃ¸rn). Skandynawski ebbe jest również wariantem Imienia (Eben, angielski, hebrajski I). Walijski ebbe jest również ulubioną formą imienia (Eberhard) niemieckim, a także niemieckim wariantem imienia (Everard dutch And). Wymowa englishts to EH-Bah w języku niemieckim â. Ebbe jest w dużej mierze używany w językach skandynawskich i niemieckich i wywodzi się ze staronordyckiego i germańskiego pochodzenia. Pochodzenie germańskie, zastosowanie w języku niemieckim: nazwa powstała jako krótka forma różnych germańskich nazw złożonych rozpoczynający…

*Werdykt:* odpowiadalne=nie, polszczyzna=nie, sensowne=nie, zbyt_ogólne=nie; główny problem: `tlumaczenie`
