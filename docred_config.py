START_E1 = '[E1]'
END_E1 = '[/E1]'
START_E2 = '[E2]'
END_E2 = '[/E2]'

SPECIAL_TOKENS = [START_E1, END_E1, START_E2, END_E2]

CLASS_MAPPING = {'headquarters location': {'id': 'P159', 'e1_type': 'ORG', 'e2_type': 'LOC'}, 'country': {'id': 'P17', 'e1_type': 'MISC', 'e2_type': 'LOC'}, 'located in the administrative territorial entity': {'id': 'P131', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'contains administrative territorial entity': {'id': 'P150', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'country of citizenship': {'id': 'P27', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'date of birth': {'id': 'P569', 'e1_type': 'PER', 'e2_type': 'TIME'}, 'place of birth': {'id': 'P19', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'inception': {'id': 'P571', 'e1_type': 'ORG', 'e2_type': 'TIME'}, 'dissolved, abolished or demolished': {'id': 'P576', 'e1_type': 'MISC', 'e2_type': 'TIME'}, 'located in or next to body of water': {'id': 'P206', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'has part': {'id': 'P527', 'e1_type': 'ORG', 'e2_type': 'PER'}, 'member of': {'id': 'P463', 'e1_type': 'PER', 'e2_type': 'ORG'}, 'performer': {'id': 'P175', 'e1_type': 'MISC', 'e2_type': 'ORG'}, 'publication date': {'id': 'P577', 'e1_type': 'MISC', 'e2_type': 'TIME'}, 'place of death': {'id': 'P20', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'date of death': {'id': 'P570', 'e1_type': 'PER', 'e2_type': 'TIME'}, 'part of': {'id': 'P361', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'capital of': {'id': 'P1376', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'capital': {'id': 'P36', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'spouse': {'id': 'P26', 'e1_type': 'PER', 'e2_type': 'PER'}, 'mother': {'id': 'P25', 'e1_type': 'PER', 'e2_type': 'PER'}, 'father': {'id': 'P22', 'e1_type': 'PER', 'e2_type': 'PER'}, 'child': {'id': 'P40', 'e1_type': 'PER', 'e2_type': 'PER'}, 'country of origin': {'id': 'P495', 'e1_type': 'ORG', 'e2_type': 'LOC'}, 'developer': {'id': 'P178', 'e1_type': 'MISC', 'e2_type': 'ORG'}, 'platform': {'id': 'P400', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'member of political party': {'id': 'P102', 'e1_type': 'PER', 'e2_type': 'ORG'}, 'point in time': {'id': 'P585', 'e1_type': 'MISC', 'e2_type': 'TIME'}, 'location of formation': {'id': 'P740', 'e1_type': 'ORG', 'e2_type': 'LOC'}, 'record label': {'id': 'P264', 'e1_type': 'ORG', 'e2_type': 'ORG'}, 'conflict': {'id': 'P607', 'e1_type': 'ORG', 'e2_type': 'MISC'}, 'educated at': {'id': 'P69', 'e1_type': 'PER', 'e2_type': 'ORG'}, 'production company': {'id': 'P272', 'e1_type': 'MISC', 'e2_type': 'ORG'}, 'employer': {'id': 'P108', 'e1_type': 'PER', 'e2_type': 'ORG'}, 'work location': {'id': 'P937', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'military branch': {'id': 'P241', 'e1_type': 'PER', 'e2_type': 'ORG'}, 'position held': {'id': 'P39', 'e1_type': 'PER', 'e2_type': 'MISC'}, 'languages spoken, written or signed': {'id': 'P1412', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'composer': {'id': 'P86', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'participant of': {'id': 'P1344', 'e1_type': 'PER', 'e2_type': 'MISC'}, 'location': {'id': 'P276', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'lyrics by': {'id': 'P676', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'member of sports team': {'id': 'P54', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'notable work': {'id': 'P800', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'author': {'id': 'P50', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'narrative location': {'id': 'P840', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'present in work': {'id': 'P1441', 'e1_type': 'PER', 'e2_type': 'MISC'}, 'characters': {'id': 'P674', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'original network': {'id': 'P449', 'e1_type': 'MISC', 'e2_type': 'ORG'}, 'genre': {'id': 'P136', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'legislative body': {'id': 'P194', 'e1_type': 'LOC', 'e2_type': 'ORG'}, 'applies to jurisdiction': {'id': 'P1001', 'e1_type': 'ORG', 'e2_type': 'LOC'}, 'owned by': {'id': 'P127', 'e1_type': 'LOC', 'e2_type': 'ORG'}, 'located on terrain feature': {'id': 'P706', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'producer': {'id': 'P162', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'continent': {'id': 'P30', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'participant': {'id': 'P710', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'sibling': {'id': 'P3373', 'e1_type': 'PER', 'e2_type': 'PER'}, 'head of state': {'id': 'P35', 'e1_type': 'LOC', 'e2_type': 'PER'}, 'territory claimed by': {'id': 'P1336', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'award received': {'id': 'P166', 'e1_type': 'PER', 'e2_type': 'MISC'}, 'residence': {'id': 'P551', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'head of government': {'id': 'P6', 'e1_type': 'LOC', 'e2_type': 'PER'}, 'director': {'id': 'P57', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'screenwriter': {'id': 'P58', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'league': {'id': 'P118', 'e1_type': 'ORG', 'e2_type': 'ORG'}, 'mouth of the watercourse': {'id': 'P403', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'subclass of': {'id': 'P279', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'end time': {'id': 'P582', 'e1_type': 'MISC', 'e2_type': 'TIME'}, 'start time': {'id': 'P580', 'e1_type': 'MISC', 'e2_type': 'TIME'}, 'creator': {'id': 'P170', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'operator': {'id': 'P137', 'e1_type': 'MISC', 'e2_type': 'ORG'}, 'publisher': {'id': 'P123', 'e1_type': 'MISC', 'e2_type': 'ORG'}, 'followed by': {'id': 'P156', 'e1_type': 'ORG', 'e2_type': 'ORG'}, 'follows': {'id': 'P155', 'e1_type': 'ORG', 'e2_type': 'ORG'}, 'cast member': {'id': 'P161', 'e1_type': 'MISC', 'e2_type': 'PER'}, 'part of the series': {'id': 'P179', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'chairperson': {'id': 'P488', 'e1_type': 'ORG', 'e2_type': 'PER'}, 'instance of': {'id': 'P31', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'manufacturer': {'id': 'P176', 'e1_type': 'MISC', 'e2_type': 'ORG'}, 'subsidiary': {'id': 'P355', 'e1_type': 'ORG', 'e2_type': 'ORG'}, 'founded by': {'id': 'P112', 'e1_type': 'ORG', 'e2_type': 'PER'}, 'official language': {'id': 'P37', 'e1_type': 'LOC', 'e2_type': 'MISC'}, 'ethnic group': {'id': 'P172', 'e1_type': 'PER', 'e2_type': 'LOC'}, 'unemployment rate': {'id': 'P1198', 'e1_type': 'LOC', 'e2_type': 'NUM'}, 'influenced by': {'id': 'P737', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'original language of performance work': {'id': 'P364', 'e1_type': 'MISC', 'e2_type': 'LOC'}, 'religion': {'id': 'P140', 'e1_type': 'PER', 'e2_type': 'MISC'}, 'basin country': {'id': 'P205', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'parent organization': {'id': 'P749', 'e1_type': 'ORG', 'e2_type': 'ORG'}, 'product or material produced': {'id': 'P1056', 'e1_type': 'ORG', 'e2_type': 'MISC'}, 'replaces': {'id': 'P1365', 'e1_type': 'ORG', 'e2_type': 'ORG'}, 'parent taxon': {'id': 'P171', 'e1_type': 'MISC', 'e2_type': 'MISC'}, 'replaced by': {'id': 'P1366', 'e1_type': 'ORG', 'e2_type': 'ORG'}, 'separated from': {'id': 'P807', 'e1_type': 'LOC', 'e2_type': 'LOC'}, 'twinned administrative body': {'id': 'P190', 'e1_type': 'LOC', 'e2_type': 'LOC'}}

DEV_TITLES = ["Lark Force", "ABBA Live", "Skai TV", "Washington Place (West Virginia)", "IBM Research – Brazil", "Lookin Ass", "Conrad O. Johnson", "Samsung Galaxy S series", "Ned McEvoy", "Portland Golf Club", "Urgut", "Robert K. Huntington", "More (The Sisters of Mercy song)", "Delaware General Assembly", "Allen County, Ohio", "Palestinian National Theatre", "Taiwanese expatriates in Vietnam", "Allen F. Moore", "Johan Gottlieb Gahn", "Siberia Governorate", "Quokka", "Dollar General", "Memogate (Pakistan)", "Live in New York (Laurie Anderson album)", "Norvelt, Pennsylvania", "Julian Reinard", "Beibu Gulf Economic Rim", "Hooshang Seyhoun", "Albemarle County, Virginia", "Kurt Tucholsky", "The Yorkshire Post", "Addy Lee", "Wilhelm Lanzky-Otto", "Jeanne Thérèse du Han", "(I Am) The Seeker", "Queen of Housewives", "Old Loggers Path", "Grand Wing Servo-Tech", "Paul Morphy", "Low Pass, Oregon", "Outline of South Sudan", "Pointr", "Villanova Preparatory School", "America's Sweetheart (album)", "Brooks Pharmacy", "Crouching Tiger, Hidden Dragon: Sword of Destiny", "Bicycles &amp; Tricycles", "Each Time You Break My Heart", "UNESCO Confucius Prize for Literacy", "Chachalaca", "CBBC", "James De Alwis", "Gershonites", "Antony Noghès", "Rose Porteous", "Think Bike", "Penedo da Saudade", "Will Weng", "Lavaca Bay", "Angleton High School", "Wellywood", "Entreat", "Jimmy Frise", "Metropolitan statistical area", "La prima notte di quiete", "Loopline Bridge", "Pokémon (anime)", "The Swingle Singers", "Rogaland County Municipality", "Pedro León Gallo", "Manon Balletti", "Achilles Last Stand", "Olof Mörck", "Taft–Katsura agreement", "ROKETSAN", "Samuel C. Brightman", "Tonie Marshall", "Washington v. Texas", "Elias Brown", "Morogoro Region", "Book of Royal Degrees", "El Tren de los Momentos", "Crisis: Behind a Presidential Commitment", "Won-yong", "Loud Tour", "Parvathy Jayaram", "Solingen", "Darren Davies (Welsh footballer)", "Ali Kuli Khan Khattak", "Riksakten", "Dan Sterling", "Guillermo Endara", "Wolfgang Thüne", "Joseph Octave Mousseau", "Bantustan", "Frederick Chiluba", "Kerstin Thorborg", "Isle of Palms, South Carolina", "Joe Gates", "WUIS", "Foulsyke", "Mark Harmsworth", "Malpai Borderlands", "US Airways Group", "Surreal (song)", "Oecophora bractella", "Cassels Lake", "East Asia Economic Caucus", "Australia–Chile Free Trade Agreement", "Susan Pharr", "Chauncey B. Brewster", "John Schofield (VC)", "Sara Black", "Overseas Press Club", "Energy and Environmental Security Initiative", "Stanisław Tymiński", "University of Uyo", "Zabiele, Warmian-Masurian Voivodeship", "Ten Commandments for Drivers", "Philipp Brammer", "LEC billing", "Toyota Avalon", "Dahod", "Kazimierz Szosland", "National Executive Committee of the African National Congress", "Resident Evil: Degeneration", "Zakhary Lyapunov", "Barry University School of Podiatric Medicine", "Nikita Bogoslovsky", "Culture of Los Angeles", "Estêvão Gomes", "United States Health Care Reform: Progress to Date and Next Steps", "Oberliga Rheinland-Pfalz/Saar", "Abdul Jabar Sabet", "École des officiers de la gendarmerie nationale", "Avery Fisher Career Grant", "Vadasserikara", "Pantabangan–Carranglan Watershed Forest Reserve", "Exe Estuary", "Edmund Hlawka", "The Killers (Bukowski short story)", "The G-String Murders", "Appenzell (village)", "American Airlines Group", "Burns Verkaufen der Kraftwerk", "Western Ukraine", "Blank Page", "Luan Bo", "Newsnight", "Leone Marucci", "Samarinda", "Brother Man", "Downy", "Nikos Aliagas", "Beyond Good &amp; Evil (video game)", "Nicky Ladanowski", "Ljiljana Raičević", "Link River", "Young Wild Things Tour", "Guido Bonatti", "Catholic Church in the Maldives", "Edward Rowan Finnegan", "History of Medicine Society", "Drum Boogie", "Boljoon", "Robert Burns Fellowship", "Northern Territory Force", "New Haven Harbor", "Mikheil Javakhishvili", "Antisemitic canard", "Klassics with a &quot;K&quot;", "Gordon Persons", "Between Five and Seven", "Dan Chupong", "Vaanathaippola", "Queen of Mauritius", "Durán, Ecuador", "National Integrated Ballistic Information Network", "Joyce Godenzi", "Santa Elena de Uairén", "Franz Wilhelm Seiwert", "Gregorio Pacheco", "M. C. Veerabahu Pillai", "Liverpool Neurological Infectious Diseases Course", "Battle of Chiari", "Woodlawn, Baltimore County, Maryland", "Little Mahantango Creek", "Tomcats Screaming Outside", "Wagner–Rogers Bill", "Motijheel Thana", "Burseraceae", "The Two Mrs. Nahasapeemapetilons", "Pleistocene coyote", "Kirnitzschtal tramway", "Gwarn Music", "Auguste Dreyfus", "Georg Jochmann", "Paul Desmarais Jr.", "Sangiran", "George Washington's resignation as commander-in-chief", "Eivind Bolle", "Bookbinder soup", "Lampedusa", "Perdono", "Word to the Mutha!", "Juan Balboa Boneke", "Isa, Nigeria", "The Christian Manifesto", "Lake Hiawatha", "The Scribe (film)", "National Flag Square", "Ninety-Two Resolutions", "Cimatti", "Symphony Hall, Boston", "The Hub (band)", "Holmenkollen Ski Museum", "Westmere, New York", "Drossinis Museum", "Nazim Khaled", "Philippine Commission", "Grant Green Jr.", "Crazy Town", "The Guianas", "Russification of Finland", "Joseph Alexander Cooper", "Thirteens (album)", "Robert Moevs", "Northern bald ibis", "O. W. Coburn School of Law", "James McLamore", "Emiliano Esono Michá", "Ko Chang-seok", "Elwood Buchanan", "Frederick August Wenderoth", "Consort Mei", "United States Ambassadors appointed by Donald Trump", "Mariana Cook", "Gei Zantzinger", "Tyche", "Powerglide (album)", "Usain Bolt Sports Complex", "Edmund Burke", "TMF (UK &amp; Ireland)", "Jefferson Madeira", "Aleksandr Alekseevich Borovkov", "I, Frankenstein", "Charles Quef", "Life in Color", "Alecu Russo", "Borderlands (series)", "Cry for Help", "Overbrook High School (New Jersey)", "Marcial Maciel", "Alfred and Plantagenet", "Jan Betley", "Lombardia (wine)", "Fatih Terim", "Heinrich LXXII, Prince Reuss of Lobenstein and Ebersdorf", "Abdullah I Al-Sabah", "Microserfs", "FLAMA", "University College London Hospitals NHS Foundation Trust", "Bigpoint Games", "Metacomet Ridge", "Great South Australian Coastal Upwelling System", "Brigden, Ontario", "I Think We're Alone Now", "Cassin's finch", "Shneur Zalman of Liadi", "List of Chinese administrative divisions by ethnic group", "Muir Beach, California", "Kokumin Dōmei", "Soccer Academy", "L'Oblat", "Larry King Live", "Patriot Hills Base Camp", "John H. Furse", "Yuriy Lutsenko", "Holden New Zealand", "Christie Elliott", "Kungliga Hovkapellet", "Nonconformist", "Kiss Each Other Clean", "First Battle of Homs", "Electoral district of Frankston East", "Sonic the Hedgehog", "Admission to the Union", "Railroad Revival Tour", "Matagorda Bay", "List of United States Armed Forces unit mottoes", "Memories (Elvis Presley song)", "Franck Piccard", "Chuck Domanico", "Fyns Hoved", "Orongorongo River", "Skylake (microarchitecture)", "Independent Democratic Action", "Ko Joo-yeon", "Joseph in Islam", "American Theocracy", "John of Islay, Earl of Ross", "Dieter Eppler", "Parks in Greater St. Louis", "The Eminem Show", "Ronald Leonard", "List of Paraguayan women writers", "A Bridge Too Far (book)", "Edward Lowassa", "Steam (Peter Gabriel song)", "Pljeskavica", "Le Dep", "Orange (India)", "Magic Eye (TV series)", "John Henry Dearle", "Joh Keun-shik", "Durgada, East Godavari district", "Canton of Raetia", "The Wedding Dance", "Sachy (writer)", "Mountain railways of India", "The Catholic Catechism (Hardon)", "Joseph Whipp", "Long Hard Road Out of Hell", "Fatima Jinnah Park", "San Pablito, Puebla", "West Bank Story", "North American Agreement on Environmental Cooperation", "Jon A. Lund", "House of Orléans", "Joshua Cushman", "Tiana (Disney)", "Route Army", "Liangzi Lake", "Music of the American Civil War", "Sidney Peel", "Hidhir Hasbiallah", "Ramey Idriss", "Carol II of Romania", "Rob James (singer)", "Cloughton", "Pinal Peak", "Andrew Fastow", "New Caledonian barrier reef", "Andrea Lilio", "Stars and Stripes Forever (film)", "My Red Hot Car", "N. C. Wyeth", "Historiographic metafiction", "Greatest Hits (Queen album)", "Revulsion (Star Trek: Voyager)", "The Murders in the Rue Morgue", "Song of Freedom", "Political positions of Donald Trump", "Joseph Israel", "Tōkai region", "Lough Owel", "Yandina, Solomon Islands", "Juan Carlos Muñoz", "Cove Lake State Park", "Battle of Sio", "Big Muddy Creek (Missouri River tributary)", "Charles Guiteau (song)", "Tomasz Bohdanowicz-Dworzecki", "William Hepburn Armstrong", "Toulouse Business School", "Cello Suites (Bach)", "Jonathan Sayeed", "Hohenwald, Tennessee", "Laurentides (electoral district)", "ELAM (Latin American School of Medicine) Cuba", "Latourell Falls", "Osmund Ueland", "Lost Verizon", "Heikki H. Herlin", "Portugal no coração", "Philip Sadée", "Fyllingen Fotball", "Across the Black Waters", "Township High School District", "Bill Frist", "Nazar Mohammad", "David Draiman", "Financial District, Manhattan", "List of Spanish inventions and discoveries", "Mike Padden", "John R. Thurman", "Björn von der Esch", "List of National Football League quarterback playoff records", "Economic history of Cambodia", "Hewanorra International Airport", "Bear Valley Springs, California", "Shin (band)", "Orange Grove affair", "Prinzenpalais, Oldenburg", "Olympic Gold (video game)", "Bengals–Browns rivalry", "Google Springboard", "Bessang Pass Natural Monument", "Togoland Campaign", "Charles W. Chesnutt", "Cranford Agreement", "Concordia Station", "Glory and Gore", "Boulevard des Capucines", "Telecom Business School", "Pangaea Ultima", "Henrik Angell", "WACP", "Virtus Pallacanestro Bologna", "Military Communications and Electronics Museum", "Chapman Square", "Alice Bunker Stockham", "Yuri Ushakov", "Teardrops (George Harrison song)", "The Game (dice game)", "David Swinford", "Aimé Millet", "Vesper sparrow", "Your Disco Needs You", "Trail Smelter dispute", "Gromshin Heights", "Ire Works", "All Together Now (The Farm song)", "Regal Mountain", "...Nothing Like the Sun", "The Hurting", "Bound Brook (Raritan River tributary)", "Delphi Greenlaw", "David Bowie filmography", "Treaty of Edinburgh–Northampton", "Le Ventre de Paris", "Tigran Petrosian", "Oliver &amp; Company", "Tropic of Cancer (film)", "Crupet", "Rachel Perry (artist)", "Nexus Q", "Zamoyski Palace", "Waterloo Moraine", "Shire of Murray", "The Virgin and Child with St Anne and St John the Baptist", "Chrysostom Blashkevich", "Ngoako Ramatlhodi", "Kyoto Imperial Palace", "Esprit Orchestra", "Anna Caselberg", "Noel Nicola", "U.S. European Command State Partnership Program", "Młociny metro station", "Wind &amp; Wuthering Tour", "Des Plaines River", "List of Presidents of Ethiopia", "Denali National Park Improvement Act", "Demonstration (Tinie Tempah album)", "Velislai biblia picta", "Medan, son of Abraham", "German torpedo boat Kondor", "Suikoden Tactics", "James P. Maher", "Kriegers Flak", "Afonso, Prince Imperial of Brazil", "United States Marine Band", "Rebelde (album)", "Greatest Hits (A-Teens album)", "Hvidovre Municipality", "Camp Caves", "Francesco De Masi", "Extreme Makeover: Home Edition", "Committee against the Finnish White Terror", "Olympic National Park", "Todas as Ruas do Amor", "ProSieben", "Titanfall: Assault", "The Private Life of Helen of Troy", "Schizoanalysis", "Contact Group (Balkans)", "They Shoot Horses, Don't They? (film)", "The Merchants of Bollywood", "Vineeth Sreenivasan", "How to Save a Life (Grey's Anatomy)", "Pete Murray", "Beaverton, Oregon", "Berge Meere und Giganten", "Howard the Duck (film)", "Ages of consent in Oceania", "X-Men Legends II: Rise of Apocalypse", "Safdar Jung (film)", "Bill Warner (writer)", "Odessa (Bee Gees album)", "Peace on Earth/Little Drummer Boy", "Uhuru Gardens", "Sappy Records", "Altsys", "Il regalo più grande", "George Nostrand", "Giuseppe Barattolo", "Christian worship", "Genc Ruli", "Inner Life", "Michael Giacchino", "Miguel Hidalgo y Costilla", "Eclipse (Meyer novel)", "Ernst Anton Nicolai", "Thomas Marlow", "Marc Savoy", "Central Massachusetts", "List of Chief Ministers of Madhya Pradesh", "Typhoon Vicente", "Triple-threat man", "Waldshut station", "Jane Livingston", "Ștefan Grigorie", "Munghana Lonene FM", "Soggy biscuit", "Lloyd Fredendall", "Peadar Bracken", "Foix (river)", "Vanya Mishra", "Boltenhagen", "Calueque", "Chicualacuala District", "Winnebago War", "Hope Town", "Georgian Foundation for Strategic and International Studies", "Everard Butler", "Wayne Gretzky Drive", "María de Buenos Aires", "Hollins University", "Sibiu", "Charles S. Johnson", "Silvan Elves", "Uptoi Village", "Suikerbosrand Nature Reserve", "CKNL-FM", "Argentine heavy metal", "Battle of Samugarh", "José María", "Sims-class destroyer", "List of Celtic F.C. records and statistics", "Phyllonorycter issikii", "Léon la lune", "Master Chief (Halo)", "Paul Beliën", "Marselisborg Gymnasium", "Dieudonné Gnammankou", "Baunilha e chocolate", "Walter Krüger (SS general)", "Two Doors Down", "United States Naval Forces Germany", "Malolos", "Józef Piłsudski Monument, Warsaw", "Ellerbusch Site", "French Resistance", "Die Haut", "List of Property Brothers episodes", "David Bohigian", "Patrick Stettner", "Rudolf Halin", "David Chipperfield", "Carlos Pedevilla", "War of the Quadruple Alliance", "George Prévost", "Cranks and Shadows", "Pete Ham", "Mitchell, Illinois", "B. V. Sreekantan", "Drake Landing Solar Community", "David Hackett", "Jephté", "Scimitar oryx", "The Answer Is Never", "Ost Autobahn", "Alex Hardcastle", "ITS launch vehicle", "Solar Jetman", "Roald", "Lego Marvel's Avengers", "Vicente Genaro de Quesada", "Tullgarn Palace", "Amandeep Khare", "Durham Bulls", "Cine-Allianz", "Pskhu", "Il Gobbo di Rialto", "Anna Karenina", "Sunol Valley", "Law of the Cayman Islands", "Henri Guédon", "Bill DeMott", "Rachel Proctor", "London Calling", "Tire, İzmir", "Google Chrome", "Michael Claassens", "William César de Oliveira", "Gatineau Olympiques", "Yi Seok", "Cleopatra's Needle", "Lemany, Warmian-Masurian Voivodeship", "Truth in Music Advertising", "Louis Lombardi", "Louise Faure-Favier", "Jared Allen", "Subramaniapuram", "Mess of Blues (Jeff Healey album)", "Lewis of Luxembourg", "Lumia imaging apps", "Martin Ulrich", "Radu Lecca", "WHTV", "Khooni Khanjar", "Sarah Gibb", "Irish experiment", "Oakland/Troy Airport", "Iowa Telecom", "Mykhaylo Fomenko", "Mikhail Kogan", "Bloomberg L.P.", "Ecuadorian Constituent Assembly", "John Anderson Moore", "Han Lei", "Municipal elections in Canada", "Tudor Chirilă", "Walter Newman (screenwriter)", "Toufique Imrose Khalidi", "Ali Abdullah Ahmed", "Great Belt Fixed Link", "Flag of Prussia", "Lehmann Bernheimer", "George Kretsinger", "Elbląg County", "Mehmet Çetingöz", "Fyodor Kulakov", "Clandestine literature", "Elliott Arnold", "Los Alerces National Park", "Alberto Valenzuela Llanos", "Kung Ako'y Iiwan Mo", "Mohammed Abdel Wahab", "King Louie", "Zarir", "Fingerpori", "Ici Paris", "Smoke Break", "Operation Unified Resolve", "I Knew You Were Trouble", "Horst Eidenmüller", "Friends of Peter G.", "List of Square Enix video games", "Ispendje", "Scotch and soda (magic trick)", "Ramblin' on My Mind", "Adolfo Nicolás", "Outright Distribution", "The Sacramento Bee", "God's Son (album)", "Eleazar Lipsky", "Brazil–Pakistan relations", "Chopin and his Europe", "Mark McNamara", "Cambodia–Vietnam Friendship Monument", "Georg Riedel (jazz musician)", "Éamon Ó Cuív", "LATAM Brasil", "Anthony Steel (actor)", "Gary Anderson (placekicker)", "Eva Bosáková", "Trane's Blues", "Rafail Levitsky", "Medini Ray", "Vase de Noces", "Michael Imperioli", "Gujranwala", "Panama during World War II", "Anatoly Chubais", "Pavol Sedlák", "Devichandraguptam", "Rambaan", "SS Keno", "Robert Taylor (computer scientist)", "Joan Burton", "Astana International Financial Centre", "Ernesto D'Alessio", "Guarenas Cathedral", "Harbour Esplanade, Docklands", "Volcanoes Stadium", "Gloria Estefan albums discography", "General Lavalle", "Number Ones (video)", "March of the Volunteers", "Asian Games", "Foundling Museum", "Olesno County", "Hang Tuah station", "Europafilm", "Tippy Walker", "Paul R. Ehrlich", "The Fortunate Pilgrim", "Gyles Longley", "Sharpbelly", "Front of Islamic Revolution Stability", "Claiborne County, Mississippi", "Scotia Plate", "Starin's Glen Island", "Hernán Caputto", "Clear and Present Danger (film)", "Fedor Ozep", "Njangalude Veettile Athidhikal", "Granskär", "Bonnie Blair", "Space Mirror Memorial", "King Haakon Bay", "Ulises Humala", "University (album)", "Raimundo Fernández-Villaverde, Marquis of Pozo Rubio", "Belmopan", "Jade (Mortal Kombat)", "Betty Bowes", "Street Fighter X Mega Man", "Christine Razanamahasoa", "Naomi van As", "Tuzantla", "Eden Games", "War between Armenia and Iberia", "Mega Man Zero", "Intelligent design", "The Expendables (film series)", "Cinema Verity", "Goodooga, New South Wales", "List of Prime Ministers of Thailand", "Bernard Cribbins", "Black Mirror (song)", "Rhodesian Bush War", "Rockefeller Brothers Fund", "Penn Line", "Laytongku", "Black Lake (Michigan)", "Osaka Bay", "Vladimir Mitrofanovich Orlov", "Willi Schneider (skeleton racer)", "Ooredoo", "Delia Gallagher", "Enasa", "Can't Take My Eyes Off You", "Torrejonian", "Jonathan Joss", "Excitebots: Trick Racing", "Schiedea haleakalensis", "John Eudes", "Three Lions", "Francisco Asenjo Barbieri", "Christian Hee Hwass", "Bill Dare", "Abbas Kiarostami", "Michel Temer", "Jerry Steiner", "Surf's Up (film)", "Christoph Daum", "Daniel Ajayi-Adeniran", "Carl Dix", "Genocide of indigenous peoples", "Kalinga (Mahabharata)", "Puskás", "Carey Business School", "Lappeenranta", "White Light Rock &amp; Roll Review", "Pierre Seghers", "The Longest Daycare", "Omero Mumba", "Cesare Bertolla", "Aramis Ayala", "Amos Hochstein", "Naomi (wrestler)", "Alan Jones (radio broadcaster)", "Independence Pass (Colorado)", "Berlin-Hohenschönhausen Memorial", "Arjuna", "Shanghainese", "Booker T. Washington", "Shiba Tōshō-gū", "Mono (software)", "Maybe (Jay Sean song)", "Minako Nishiyama", "Shihlin Electric", "Ali Akbar Moradi", "Leandro Mbomio Nsue", "Lee Hall Mansion", "Clark Lake (Gogebic County, Michigan)", "Antioch", "Myra Clark Gaines", "Christopher Franke", "Heeney, Colorado", "Derby, Connecticut", "Zevs (artist)", "Glandularia", "In a Silent Way", "The Reverent Wooing of Archibald", "Kiirunavaara", "Lisa Mona Lisa", "Rock n Roll (Avril Lavigne song)", "P. D. T. Acharya", "Francis Pipe-Wolferstan", "Watts Station", "List of colleges and universities in Connecticut", "Luis Roche", "Jirō Shiizaki", "Breakout (video game)", "Operator Toll Dialing", "Mola di Bari", "Carl Buchheister", "Alexeni", "Paradise Cracked", "Eleonora Vallone", "List of regional railway stations in Victoria", "Cesare Mori", "In the Mood", "Navy, Army and Air Force Institutes", "Treaty of Ghent", "Microsoft Sync Framework", "Museum of Army Flying", "Tulelake camp", "List of songs recorded by Rage Against the Machine", "Goght", "Malo (saint)", "O'Shaughnessy Dam (Ohio)", "List of Major League Baseball first-round draft picks", "Pittsburgh Catholic", "The Colts (vocal group)", "Orangina", "Malay houses", "Justinian Tamusuza", "Aspren", "IFK Norrköping", "Vätsäri Wilderness Area", "List of diplomats of Norway to China", "Corps Hubertia Freiburg", "Fugue", "The Horn Blows at Midnight", "Mavis Grind", "Rufus Carter", "While the City Sleeps, We Rule the Streets", "Peters Club", "Mehdi Karroubi", "Bohumín", "What a Time to Be Alive", "Association of Australian Artistes", "Milton Friedman Institute for Research in Economics", "Gambier Island", "Tartu", "Babyfingers", "List of longest rivers of Canada", "First Gallagher Ministry", "Snoopy and His Friends", "Aino (mythology)", "Bone Machine", "Hélé Béji", "List of islands of Western Australia, M–Q", "Ripton (fictional town)", "Asian Men's Volleyball Championship", "How Could Hell Be Any Worse?", "Irish in the British Armed Forces", "The Legend of Zelda: The Minish Cap", "Tourism in Montreal", "Ferrous metallurgy", "Li Jiancheng", "Louis Chollet", "Dalma (island)", "William James Wallace", "Raunaq (album)", "The Storming of the Winter Palace", "Matthew de Glendonwyn", "PLATIT", "Ruth Winona Tao", "Dante Alighieri Society", "House of Angels", "Kusha (Ramayana)", "Velocifero", "Agustin Perdices", "Reddevitz Höft", "Hutchinson Commons", "South Wigston High School", "María la del Barrio", "Distant Earth", "Sean Spencer", "Rio Grande de Mindanao", "Kitigan Zibi", "Ramada (shelter)", "Something to Believe In (Poison song)", "Ross Alger", "G. V. Belyi", "Baltimore and Ohio Railroad Martinsburg Shops", "Bombtrack", "USS Lyndon B. Johnson", "Henri de Boulainvilliers", "Neue Bachgesellschaft", "John Alexander Boyd", "Joy Division discography", "Song Nation", "Frédéric Talgorn", "Piton des Neiges", "Chapel of St. Francis of Assisi (Esperanza Base)", "The Crazy World of Arthur Brown (album)", "Ocean Waves (film)", "The Mudlark", "João Paulo de Silva", "Assassin's Creed Unity", "Briggs Terrace", "Atlantic Charter", "Asus VivoTab", "Viscount Alanbrooke", "House of Hanover", "Load (album)", "Raymond Hermantier", "Religious education in Romania", "Bad Astronaut", "Leonessa", "Hong Kong Chinese official football team", "Pettigrew State Park", "Scott Porter", "This Little Girl of Mine", "Bambi II", "Ålgård Line", "Blue River (Colorado)", "Michael Scanlon", "Vladislav Frolov", "Joseph R. Anderson", "Sycamore Canyon Wilderness", "The Archbishop", "Enterprise Objects Framework", "Battle of Kashii", "Kahului Airport", "Lunula (amulet)", "Anne-Marguerite Petit du Noyer", "Dustin's Bar Mitzvah", "Janaka", "Ulysses (novel)", "Arthur Percival", "National Turkey Federation", "Henry Halleck", "John Ripley (USMC)", "Pierre Le Gros the Younger", "Oleg Tinkov", "Henri de Buade", "Upper Ammonoosuc River", "Zillebeke Churchyard Commonwealth War Graves Commission Cemetery", "List of composers of Caribbean descent", "Andy Cole", "Bajofondo", "Souvlaki", "Cy Becker, Edmonton", "Amambanda", "Mount Shinn", "Imam khatib (Sunni Islam)", "John D. McWilliams", "Statue of Jan Smuts, Parliament Square", "Lucien L'Allier", "Christian Atsu", "Ernst-Ludwig Schwandner", "Angry Candy", "Anthony G. Brown", "Extraordinary Merry Christmas", "Çağlar Söyüncü", "Faliero coup", "Cornelis Ketel", "Trevor Brooking", "Ne crois pas", "Arcadia (band)", "Mistborn", "Anton Erhard Martinelli", "The Sound Barrier", "Panlongcheng", "Laurence M. Keitt", "Ahmet Kaya", "Brookston, Minnesota", "Georgi Dimitrov Mausoleum", "TY.O", "Mendenhall Glacier", "Signaling of the New York City Subway", "Beijing Ducks", "Mount Garibaldi", "Guybrush Threepwood", "South Gondar Zone", "Foots Creek, Oregon", "Reichskommissariat of Belgium and Northern France", "Phineas Clanton", "Vauban (train)", "Piracy on Falcon Lake", "Coolidge Cricket Ground", "Liang Congjie", "The Time of the Doves", "White Sea", "Perpetual virginity of Mary", "Church of Saint Francis of Assisi (Ouro Preto)", "Johannes Cuspinian", "Iwo Byczewski", "Royal Arsenal", "Ramapo High School (New Jersey)", "Essingen Islands", "Soldier (Gavin DeGraw song)", "Paul Pfeifer"]