# Muestras — Task 2.1

Prompt: `ROMEO:`. Cada salida contiene el prompt y 200 caracteres nuevos.

## Configuración A
Estrategia: `greedy`; parámetros: `{}`.

### Muestra A1
```text
ROMEO:
It the shall the son the she son the son the sone
The son the shen the son the she son the son the she
The son the she shere son the sourt the sourterer
The sore the soull the soull the son the sourt
```

### Muestra A2
```text
ROMEO:
It the shall the son the she son the son the sone
The son the shen the son the she son the son the she
The son the she shere son the sourt the sourterer
The sore the soull the soull the son the sourt
```

### Muestra A3
```text
ROMEO:
It the shall the son the she son the son the sone
The son the shen the son the she son the son the she
The son the she shere son the sourt the sourterer
The sore the soull the soull the son the sourt
```

### Muestra A4
```text
ROMEO:
It the shall the son the she son the son the sone
The son the shen the son the she son the son the she
The son the she shere son the sourt the sourterer
The sore the soull the soull the son the sourt
```

### Muestra A5
```text
ROMEO:
It the shall the son the she son the son the sone
The son the shen the son the she son the son the she
The son the she shere son the sourt the sourterer
The sore the soull the soull the son the sourt
```

## Configuración B
Estrategia: `top_k`; parámetros: `{'k': 5, 'tau': 0.7}`.

### Muestra B1
```text
ROMEO:
So the seay that should so the hinge the that will steang
To to the sent and here the bat hand sold
Thee seat the the deadt his his haven this sore,
That with show soull mow the deathers the there sh
```

### Muestra B2
```text
ROMEO:
Why I shat heard word shere hear him.


LORENENG
BARD IV:
Where nowerer will shay heave so hat his will of there
Anked he the mange stinds heat thou murereries
The mure sore shour as thar my asheres,
```

### Muestra B3
```text
ROMEO:
It whis his, seel somer sall the shis soll sont your trathe
Breen thar shat havering that the the werence
To will somen the wore with shingh an and truche
The do to shall my heess whond and sherere
T
```

### Muestra B4
```text
ROMEO:
It say with that my songe be and wand hour his tre
Ther dear to the heant a hith this will sering,
Ind this theer wer and ald an
Fither sall to be thor hat the hour masere son han
He sollown the the 
```

### Muestra B5
```text
ROMEO:
I shon him haven ware to hat the say, and and sthere
Thisen are a sharenges shall sean havend and him.

COLIORD IICK:
Wellle and the mid thall his we thor suell hivess
What sold shere to deert an the
```

## Configuración C
Estrategia: `top_k`; parámetros: `{'k': 50, 'tau': 1.0}`.

### Muestra C1
```text
ROMEO:
Iway clat havensemm, Faflud prlovers ick: her lend
Blikse diest to sotn feordfors and maD
Rore, dir miress of droor siry to the ho;--for fiturer!
Sechinh in of pith thate lill cheseromminser;
Gard I 
```

### Muestra C2
```text
ROMEO:
Your stat not lif lick nies unok's.

CLAND BENTARLAR:
By nat with hak surnd far.
Gore uppoand's not wyell'd it demery;
Ongher.

SIV:
An sell.

HLARD VIS:
May, sherelel him sulls you
Haty searrelt! Bu
```

### Muestra C3
```text
ROMEO:
Brow't wheser, priceer cot. Rickilks sabak day
Unow and swithencen tathbin'd sheant ryou rat ot of ally
Thest hat amy his of my, guinghelershad my pen gow
wor ind hort a bacher eepongh'd 'thermish:
B
```

### Muestra C4
```text
ROMEO:
DUR Owimby I couch.
ELOY USBE:
Urkernseriout, my the Leres dent, ande;
Stit arrn andy sitho so haton ofnes,
Weat war may come frot butt?
Fore I Lieccand Cfom,
And neter your End qur ad menorthertoy r
```

### Muestra C5
```text
ROMEO:
Yo?

QUKENRIO:
Ay Of, my sheme may wity myishmed whaty chopmjer.
Thatrecy herceg and year'llfiear sparrffords and,
Houre hano? Nore!

O, mairh you doudstond benquse pingess sucare of chad?
Touke I ch
```

## Configuración D
Estrategia: `top_p`; parámetros: `{'p': 0.7, 'tau': 0.7}`.

### Muestra D1
```text
ROMEO:
That say a and shall have mant son sull see the anderers
To sure and may be strord thath and beared with to the thee
That be me sucher the walll he be and of his soulle
That my bet my the soller of s
```

### Muestra D2
```text
ROMEO:
She whe that wat see a monte of the my dore.

KINGHARD RII:
Here math ast she searte fair the the dist heart and her
Meard a do herer that werering a shir his
For his heat here in and has my and so b
```

### Muestra D3
```text
ROMEO:
What sord the dough and a do ther hant be were forthion
That heat the shee he the broth bearter,
What you det my dent sond in of whish of the
Buthereres to be sherir sore the fath the with the therer
```

### Muestra D4
```text
ROMEO:
Your hime is the hime is and the the and the sone
I the son that shat his comers, and that hat hearther
The say with and sen the frore the me here with werer in mast
The the will in bet his fort ther
```

### Muestra D5
```text
ROMEO:
By the have dist bealt in and the with the be sont,
And a freath sont his sean all the be so that mand
He me that mand the the deant and so hand and
The me me and of his my and and and there hist
And
```

## Configuración E
Estrategia: `top_p`; parámetros: `{'p': 0.95, 'tau': 1.0}`.

### Muestra E1
```text
ROMEO:
He your mod beter. Ror sirly;
Well when sateils rald meembersering in hat is ofe.
Rouct enrer opo in, meerr an for wall low hant and
Sores inn toon arthers so survel than not I I
And:
Tho tith isprer
```
Traza top-p — menor núcleo: paso 1 ('
'): 1 candidatos, paso 83 ('
'): 1 candidatos; mayor núcleo: paso 21 ('R'): 31 candidatos, paso 135 ('S'): 31 candidatos.

### Muestra E2
```text
ROMEO:
If be wit hise broth the selle and in herr, mon's foored.

This my limerer:
I'e, ficht hay hids fat weall ofor this hepos
Theish ough wis with hon to gonge your the word all at famtile
Mienges is pri
```
Traza top-p — menor núcleo: paso 1 ('
'): 1 candidatos, paso 59 ('
'): 1 candidatos; mayor núcleo: paso 124 ('T'): 32 candidatos, paso 5 ('b'): 31 candidatos.

### Muestra E3
```text
ROMEO:
Snowel Ind larne, not, merall, is lin'd thean tie thee ald
Withn contust: thon, not fith them wort brod?

Com worde bataing weay the is warden thil do hamt: adrened
I sir worl is of to my morgly, in 
```
Traza top-p — menor núcleo: paso 1 ('
'): 1 candidatos, paso 19 (' '): 1 candidatos; mayor núcleo: paso 61 ('W'): 36 candidatos, paso 159 ('a'): 31 candidatos.

### Muestra E4
```text
ROMEO:
Youre in bet ut kive, a wind woull's to then sod had?

LORGERY:
Hes a rig courerce mind ale beitincort.

RINIETHARDY YE FHUMEST:
Theaw:
Thou dat ample ot rithing.
To heave benters dead to forn hass o
```
Traza top-p — menor núcleo: paso 1 ('
'): 1 candidatos, paso 55 ('
'): 1 candidatos; mayor núcleo: paso 168 ('h'): 36 candidatos, paso 70 ('a'): 31 candidatos.

### Muestra E5
```text
ROMEO:
Bit herdon:
Prey for, break loven harve the you shaulll,
Thours kar for to thinge my donted what handed,
Sis cringy but himpy ist plarererestion,
For, 'thern farden it.

RICIUSABET:
I hid to welll fo
```
Traza top-p — menor núcleo: paso 1 ('
'): 1 candidatos, paso 23 (' '): 1 candidatos; mayor núcleo: paso 153 ('''): 31 candidatos, paso 154 ('t'): 31 candidatos.
