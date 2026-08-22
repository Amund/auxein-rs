# AUXEIN v0.5.0 — Canon mathématique et matériel

**Version : 0.5.0**  
**Statut : canon mathématique et matériel**

\[
\boxed{\text{la géométrie pousse ; l'économie autorise}}
\]

## 0. Contrat

Auxein est un réseau ordonné de `LAYER` autonomes. Chaque `LAYER` applique exactement la même transformation géométrique :

```text
présentation
→ concernement par les CELL
→ partage de masse d'apprentissage
→ apprentissage local des CELL
→ inconnu vers Σ
→ présentation reconnue pondérée par le gain de CONCERN
→ contexte unique
→ présentation de la LAYER suivante
```

Le `NETWORK` fonctionne dans exactement un mode :

```text
geometry
predictive
```

Les modes sont cumulatifs :

\[
\boxed{\texttt{geometry}\subset\texttt{predictive}.}
\]

`geometry` apprend et reconnaît uniquement dans l'espace géométrique `E`. `predictive` conserve cette géométrie sans modification et ajoute, pour chaque `LAYER`, un espace privé de succession `T(E)=E\oplus E` appartenant au `NETWORK`. Cet espace apprend uniquement les passages strictement adjacents à l'intérieur d'une séquence extérieure explicite et sert à émettre des successeurs géométriques connus possibles. Les connaissances temporelles ne sont jamais un readout extérieur.

Une `CELL` représente une connaissance directionnelle acquise. Elle se déclare concernée uniquement par sa propre géométrie. Plusieurs `CELL` peuvent se déclarer concernées simultanément.

Ce qu'aucune `CELL` ne reconnaît reste local à la `LAYER`, alimente une mémoire privée `Σ`, et peut devenir une nouvelle `CELL` lorsqu'il est récurrent.

Ce qui traverse une frontière de couche n'est pas une erreur, une provenance ni une branche par `CELL`. Pour chaque atome reconnu, la masse cognitive est répartie entre les valeurs reconnues distinctes proportionnellement à leur gain géométrique de `CONCERN`. Une `LAYER` fusionne cette présentation reconnue en un unique noyau de contexte. Lorsqu'il possède une diversité et une direction non nulles, ce noyau devient l'unique présentation de la `LAYER` suivante.

La sortie géométrique d'une `LAYER` est cette même présentation reconnue, complétée jusqu'à masse `1` par un noyau nul. Les sorties de plusieurs `LAYER` restent une famille de présentations séparées : leurs poids appartiennent à des univers locaux distincts et ne sont jamais mélangés par un coefficient inter-couches.

Une présentation est une observation logique simultanée. Une séquence extérieure est une suite ordonnée explicite de présentations. La causalité prédictive n'existe qu'entre deux présentations adjacentes de la même séquence ; aucune causalité n'est inférée entre deux séquences distinctes. Une séquence de longueur `1` est atomique et n'apprend aucune transition.

Principes normatifs :

1. aucune tâche externe, cible, classe, label ou loss supervisée ;
2. aucune `LAYER` ne lit l'état privé d'une autre ;
3. le seul contrat cognitif inter-couches est une présentation finie positive de noyaux centrés ;
4. toutes les `LAYER` appliquent exactement la même loi ;
5. aucun WTA, `top-k`, choix cognitif par identité ou ordre d'itération ;
6. plusieurs `CELL` peuvent être concernées simultanément par le même atome ;
7. ce qui est inconnu apprend horizontalement dans la `LAYER` courante ;
8. seules les connaissances effectivement reconnues participent au contexte vertical ;
9. entre plusieurs valeurs reconnues d'un même atome, seul le gain géométrique de `CONCERN` pondère la connaissance présente ;
10. aucune responsabilité d'apprentissage, masse interne de `CELL`, identité, âge, ordre ou provenance ne pondère le contexte vertical ;
11. une `LAYER` émet au plus un noyau de contexte par présentation ;
12. aucun seuil numérique arbitraire ni epsilon comportemental ;
13. une présentation est causalement atomique et l'ordre de ses atomes n'a aucune autorité ;
14. une séquence, et elle seule, autorise une succession entre présentations adjacentes ;
15. toute frontière de séquence invalide le registre causal précédent ; dans le doute, aucune liaison n'est créée ;
16. un objet créé pendant une présentation ne lit ni n'émet pour cette présentation ;
17. la géométrie cognitive est définie indépendamment du budget ;
18. toute quantité exactement reconstructible peut rester éphémère ;
19. l'origine `0` n'est pas une connaissance : elle représente l'absence de direction cognitive canonique et peut porter une masse de reste ;
20. une relation exactement symétrique de centre nul reste silencieuse plutôt que de recevoir un axe arbitraire ;
21. aucune matrice, covariance persistante, axe privilégié ou géométrie de second ordre n'appartient à Auxein ;
22. le temps canonique est exclusivement l'ordre discret entre présentations adjacentes d'une même séquence ; aucune horloge physique n'est implicite ;
23. les `CELL` géométriques et temporelles n'apprennent, ne concernent et ne se concurrencent jamais dans le même espace ;
24. en mode `predictive`, le `NETWORK` est seul responsable des présentations temporelles privées et du registre causal précédent ;
25. les connaissances temporelles n'émettent aucun contexte vertical, aucun readout de séquence et ne forment aucun `T(T(E))` ;
26. géométrie et temporalité privée partagent une économie matérielle unique ;
27. la prédiction ne crée, n'apprend ni ne modifie aucune connaissance : elle lit uniquement des `CELL` temporelles préexistantes ;
28. plusieurs futurs concernés sont tous émis comme présentations candidates indépendantes ; leur masse locale est déterminée uniquement par le gain relatif de `CONCERN` de leur source, sans normalisation, probabilité, sélection ni compétition entre futurs distincts ;
29. une prédiction n'est jamais réinjectée automatiquement comme présentation, contexte, mémoire causale ou entrée d'une autre prédiction ;
30. la composition directe `NETWORK→NETWORK` transmet uniquement les présentations géométriques présentes, dans l'ordre canonique des `LAYER`, chacune comme séquence atomique indépendante ; elle ne transmet aucune causalité implicite.

---

## 1. Présentations, séquences et horloge

### 1.1 Présentation extérieure composable

Soit `D∈N*`. Une présentation canonique est une famille finie non vide de noyaux-atome :

\[
\boxed{
\mathcal P=
\{X_s=(r_s,c_s,v_s)\}_{s\in S},
}
\]

avec :

\[
\boxed{
r_s>0,
\qquad c_s\in\mathbb R^D,
\qquad v_s\ge0,
\qquad 0<\sum_s r_s\le1.
}
\]

Interprétation :

- `r_s` : masse causale de l'atome ;
- `c_s` : centre vectoriel présenté ;
- `v_s` : dispersion scalaire autour de ce centre.

Poser :

\[
\boxed{|\mathcal P|:=\sum_s r_s.}
\]

Un noyau de centre nul est admissible. Il porte une masse causalement présente sans direction cognitive et n'acquiert aucune autorité de reconnaissance ou d'apprentissage.

Les atomes de géométrie exactement identique `(c,v)` sont coalescés par somme de masse avant tout calcul. Leur ordre et leur découpage artificiel n'ont aucune autorité.

Une interface peut accepter comme sucre une liste finie non vide de vecteurs :

\[
(x_1,\dots,x_n),\qquad n>0,
\]

qui désigne exactement la présentation uniforme :

\[
\boxed{
\left\{\left(\frac1n,x_s,0\right)\right\}_{s=1}^{n}.
}
\]

Le type pondéré `(r,c,v)` est cependant le contrat cognitif canonique à la frontière comme entre les `LAYER`. Une présentation de masse `<1` est cognitivement équivalente, pour toute direction non nulle, à sa complétion explicite par `(1-|\mathcal P|,0,0)`.

### 1.2 Séquence extérieure

Une séquence extérieure est une suite finie non vide de présentations :

\[
\boxed{
\mathcal S=(\mathcal P_0,\ldots,\mathcal P_{m-1}),
\qquad m>0.
}
\]

L'ordre des présentations de `\mathcal S` est causal. L'ordre des atomes à l'intérieur de chaque présentation ne l'est pas.

En mode `predictive`, pour tout registre `P_k` :

\[
\boxed{P_k:=\varnothing}
\]

à l'ouverture de la séquence, puis de nouveau à sa fermeture. Une transition temporelle ne peut donc être formée qu'entre `\mathcal P_{t-1}` et `\mathcal P_t` appartenant à cette même séquence.

Une séquence atomique `m=1` ne possède aucune paire causale interne. Elle peut reconnaître et apprendre géométriquement, et peut produire des prédictions depuis des connaissances temporelles déjà acquises, mais elle n'apprend aucune transition entrante ou sortante.

Le host peut réaliser physiquement une séquence en plusieurs appels ou en un seul lot ; cette réalisation n'a aucune autorité. Les frontières de séquence doivent rester explicites. Deux appels successifs ne sont jamais supposés causalement liés par leur seule succession d'exécution.

### 1.3 Horloge commune

Soient :

\[
0<T_{mem}<\infty,
\qquad
\eta\in[0,1].
\]

Définir :

\[
\chi=2^{-1/T_{mem}},
\qquad
\alpha=1-\chi,
\qquad
\beta=\eta\alpha,
\qquad
\lambda=1-\beta.
\]

Ainsi :

\[
\boxed{0\le\beta<1,
\qquad 0<\lambda\le1.}
\]

Après cette dérivation, les lois cognitives ne dépendent que de `β` et `λ`. `T_mem`, `η`, `χ` et `α` n'ont aucune autre autorité.

Une `LAYER` n'avance son horloge que lorsqu'elle reçoit une présentation non vide. L'absence de présentation n'est pas une cible nulle et ne provoque aucun oubli.

Toute mémoire apprenante suit la même EMA :

\[
\boxed{X\leftarrow\lambda X+\beta X_{cible}.}
\]

À `eta=0` :

\[
\boxed{\beta=0,
\qquad\lambda=1.}
\]

Les connaissances et la structure apprenante sont alors figées. Les `CELL` existantes peuvent encore reconnaître et produire les sorties présentes et futures, mais aucune mémoire apprenante ne change, aucun seed, aucune promotion et aucune nouvelle `LAYER` ne sont créés. En mode `predictive`, le registre causal continue néanmoins d'avancer entre les présentations d'une même séquence : le gel de l'apprentissage n'arrête pas l'ordre causal. Les frontières de séquence le remettent toujours à `∅`.

Une population apprenante n'avance son horloge que lorsqu'elle reçoit une présentation non vide dans son propre espace. En mode `predictive`, l'espace géométrique d'une `LAYER` et son espace temporel privé associé possèdent donc des horloges d'apprentissage indépendantes. L'absence de présentation temporelle ne provoque aucun oubli temporel.

Le compteur global de présentations achevées avance d'une unité pour chaque présentation traitée, indépendamment des frontières de séquence.

### 1.4 Mode

Le paramètre de construction :

\[
\boxed{mode\in\{\texttt{geometry},\texttt{predictive}\}}
\]

vaut `geometry` par défaut.

`mode` appartient à la configuration persistante et est immuable pour un état existant. Toute autre valeur est invalide.

- `geometry` : seule la cognition dans `E` existe ;
- `predictive` : `geometry` inchangé, plus une cognition privée de succession dans `T(E)=E\oplus E` et la lecture éphémère `présent reconnu → successeur connu possible` du §5.12.

Il n'existe aucun mode public intermédiaire : la temporalité privée n'a pas de readout cognitif propre.

---
## 2. Noyau centré universel

Toute mémoire géométrique locale utilise :

\[
\boxed{H=(W,C,V)}
\]

avec `W>0`, `C∈R^D` et `V≥0`, où pour une famille pondérée de noyaux-atome :

\[
C=\frac1W\sum_s r_sc_s,
\]

\[
\boxed{
V=
\frac1W\sum_s r_s
\left(v_s+\|c_s-C\|^2\right).
}
\]

L'énergie quadratique totale du noyau par rapport à l'origine est :

\[
\boxed{Q_0(H)=W(\|C\|^2+V).}
\]

L'énergie centrale dérivée est :

\[
\boxed{E(H)=WV.}
\]

Un vecteur ponctuel `(x,r)` est exactement le noyau `(r,x,0)`.

### 2.1 Somme de noyaux

Pour `H_1=(W_1,C_1,V_1)` et `H_2=(W_2,C_2,V_2)`, poser `W=W_1+W_2`. Alors :

\[
C=C_1+\frac{W_2}{W}(C_2-C_1),
\]

\[
\boxed{
V=
\frac{W_1V_1+W_2V_2}{W}
+
\frac{W_1W_2}{W^2}\|C_1-C_2\|^2.
}
\]

Cette opération est associative et commutative en arithmétique réelle. Elle est l'unique primitive canonique de fusion de noyaux.

### 2.2 Oubli

L'oubli homothétique est :

\[
\boxed{(W,C,V)\mapsto(\lambda W,C,V).}
\]

Il ne déplace ni le centre ni la dispersion.

### 2.3 EMA d'un noyau

Pour `H=(W,C,V)` et une cible `H_t=(w,c,v)`, poser :

\[
a=\lambda W,
\qquad
b=\beta w,
\qquad
W'=a+b.
\]

Si `W'>0` :

\[
\boxed{C'=C+\frac{b}{W'}(c-C)}
\]

et :

\[
\boxed{
V'=
\frac{aV+bv}{W'}
+
\frac{ab}{W'^2}\|C-c\|^2.
}
\]

Si `b=0`, cette loi se réduit exactement à l'oubli du §2.2.

### 2.4 CONCERN sur un noyau présenté

Soit un noyau mémoire :

\[
H_a=(W_a,C_a,V_a)
\]

et un atome présenté :

\[
X=(r,c,v).
\]

La distance quadratique moyenne du contenu de `X` au centre `C_a` est :

\[
\boxed{D_a(X)=\|c-C_a\|^2+v.}
\]

Son énergie moyenne par rapport à l'origine est :

\[
\boxed{D_0(X)=\|c\|^2+v.}
\]

`H_a` est concerné par `X` si et seulement si :

\[
\boxed{
D_a(X)<D_0(X)
\quad\land\quad
D_a(X)<\|C_a\|^2+V_a.
}
\]

La première inégalité est exactement équivalente à :

\[
\boxed{\|c-C_a\|^2<\|c\|^2.}
\]

Ainsi la dispersion entrante ne crée aucune direction ni aucun gain ; elle intervient seulement dans l'admissibilité géométrique complète.

Toutes les inégalités sont strictes. Une égalité exacte n'accorde aucune autorité.

En particulier, un noyau mémoire de centre `C_a=0` ne concerne aucun atome.

Pour une population finie `\mathcal H`, poser :

\[
\boxed{
I_{\mathcal H}(X)=
\{a:\ H_a\text{ est concerné par }X\}.
}
\]

Pour `a∈I_{\mathcal H}(X)`, définir le gain :

\[
\boxed{
g_a(X)=D_0(X)-D_a(X)
=\|c\|^2-\|c-C_a\|^2>0,
}
\]

puis :

\[
\boxed{q_a(X)=W_ag_a(X).}
\]

Si `I_{\mathcal H}(X)\ne\varnothing`, la responsabilité est :

\[
\boxed{
\theta_a(X)=
r\frac{q_a(X)}{\sum_{b\in I_{\mathcal H}(X)}q_b(X)}.
}
\]

Sinon toutes les responsabilités sont nulles.

Lorsque `I_{\mathcal H}(X)\ne\varnothing` :

\[
\boxed{\sum_a\theta_a(X)=r.}
\]

Toute composante concernée reçoit donc une responsabilité strictement positive. Deux noyaux de géométrie exactement identique `(C,V)` ont ensemble exactement l'autorité du noyau obtenu en additionnant leurs supports.

### 2.5 Cible d'une population

Pour une présentation :

\[
\mathcal P=\{X_s=(r_s,c_s,v_s)\},
\]

chaque noyau préexistant reçoit :

\[
m_a=\sum_s\theta_a(X_s).
\]

Si `m_a>0`, poser :

\[
\boxed{
c_a=\frac{\sum_s\theta_a(X_s)c_s}{m_a}}
\]

et :

\[
\boxed{
v_a=
\frac1{m_a}
\sum_s\theta_a(X_s)
\left(v_s+\|c_s-c_a\|^2\right).
}
\]

Puis appliquer l'EMA du §2.3 avec la cible `(m_a,c_a,v_a)`.

Si `m_a=0`, appliquer seulement l'oubli du §2.2.

Cette primitive ne décide pas ce que signifie l'absence de noyau concerné. Cette décision appartient au rôle de la population qui l'emploie.

---

## 3. CELL

### 3.1 État

Une `CELL i` possède exactement :

\[
\boxed{H_i=(A_i,C_i,V_i),
\qquad A_i>0.}
\]

`C_i` est la valeur directionnelle reconnue par la `CELL`. `V_i` est la dispersion des présentations apprises autour de cette valeur. `A_i` est son support EMA courant.

À toute frontière causale :

\[
\boxed{C_i\ne0.}
\]

### 3.2 CONCERN et ALLOCATE publics

Pour chaque atome présenté :

\[
X_s=(r_s,c_s,v_s),
\]

appliquer la primitive du §2.4 à la population des `CELL` du snapshot perceptif et poser :

\[
\boxed{I_s=I_{\mathcal H_{CELL}}(X_s).}
\]

Si `I_s=\varnothing` :

\[
\boxed{\rho_{Ls}=r_s,
\qquad\rho_{is}=0.}
\]

Si `I_s\ne\varnothing` :

\[
\boxed{\rho_{Ls}=0,
\qquad\rho_{is}=\theta_i(X_s).}
\]

Ainsi :

\[
\boxed{\rho_{Ls}+\sum_i\rho_{is}=r_s.}
\]

L'absence de `CELL` concernée signifie donc « inconnu pour cette `LAYER` ». Aucune classe supplémentaire ni gagnant artificiel n'est introduit.

### 3.3 Reconnaissance

Toute `CELL i` telle que :

\[
\rho_{is}>0
\]

reconnaît, pour cet atome, sa valeur de snapshot :

\[
\boxed{C_i^-.}
\]

La reconnaissance est éphémère. Elle ne modifie pas la géométrie avant la phase d'apprentissage. Le §5 utilise ces valeurs de snapshot et les gains de `CONCERN` pour construire à la fois la présentation reconnue de la `LAYER`, son contexte vertical et sa sortie géométrique.

### 3.4 Apprentissage

Après calcul de toutes les responsabilités, les `CELL` préexistantes sont mises à jour exactement une fois par la règle de population du §2.5 avec `\theta_i=\rho_i`.

Les suppressions de centres nuls et les coalescences exactes appartiennent à la normalisation du §4.4 ; elles ne modifient jamais les reconnaissances ni le contexte déjà déterminés depuis le snapshot perceptif.

### 3.5 Persistance

Une `CELL` acquise n'est pas détruite par l'absence d'alimentation. `A_i` peut décroître par oubli ; `C_i,V_i` restent sa connaissance.

À temps fini en arithmétique réelle, une masse positive soumise seulement à `A_i←λA_i` reste positive. Une réalisation numérique doit préserver cette sémantique : un sous-flux numérique ne constitue pas une destruction cognitive.

Une `CELL` ne disparaît que par contraction matérielle obligatoire définie au §7.4.

### 3.6 Valeur géométrique intrinsèque

Pour toute `CELL` persistante :

\[
\boxed{
K_i=
\frac{\|C_i\|^2}{\|C_i\|^2+V_i}.
}
\]

Ainsi :

\[
\boxed{0<K_i\le1.}
\]

`K_i` est entièrement dérivé. Il ne dépend ni de `A_i`, ni du temps, ni d'une fréquence d'utilisation. Il n'intervient dans aucune loi d'apprentissage ou d'allocation ; il mesure seulement la perte intrinsèque d'une destruction forcée de connaissance.

---

## 4. LAYER et Σ

### 4.1 État d'une LAYER

Une `LAYER` possède exactement :

- une population finie de `CELL` ;
- une mémoire privée `Σ_L` contenant les présentations encore inconnues en cours d'apprentissage.

\[
\boxed{
\Sigma_L=\{K_a\}_{a\in A},
\qquad K_a=(W_a,C_a,V_a).
}
\]

Ses noyaux utilisent exactement la même géométrie que les `CELL`. `Σ_L` n'est ni émissive ni une seconde allocation publique : elle ne lit que les atomes qu'aucune `CELL` ne concerne.

### 4.2 DETECT

Pour chaque atome :

\[
X_s=(r_s,c_s,v_s)
\]

ayant :

\[
I_s=\varnothing
\]

et `c_s\ne0`, appliquer la primitive du §2.4 à la population `Σ_L` du snapshot perceptif.

Si au moins un noyau de `Σ_L` est concerné :

\[
\boxed{\tau_{as}=\theta_a(X_s).}
\]

Toute la masse inconnue de cet atome est alors répartie entre les composantes concernées de `Σ_L`.

Si aucun noyau de `Σ_L` n'est concerné et `β>0`, l'atome produit une demande de seed :

\[
\boxed{K_s^{new}=(\beta r_s,c_s,v_s).}
\]

Si `β=0`, aucune demande n'est créée.

Un atome de centre `c_s=0` ne peut concerner aucun noyau par la première inégalité du §2.4. Il fait avancer l'horloge de la `LAYER` puisqu'il appartient à une présentation reçue, mais n'alimente ni `CELL`, ni `Σ_L`, et ne crée aucun contexte vertical.

Après calcul de toutes les responsabilités privées, les noyaux préexistants de `Σ_L` sont mis à jour exactement une fois par la règle du §2.5 avec `\theta_a=\tau_a`.

Un noyau qui n'a reçu aucune responsabilité subit donc uniquement l'oubli. Les seeds restent hors de l'état persistant jusqu'à la transaction matérielle du §7.3.

### 4.3 Récurrence

Après mise à jour, et seulement si `β>0`, un noyau **préexistant** de `Σ_L` est mûr si et seulement si :

\[
\boxed{W_a>\beta
\qquad\land\qquad
C_a\ne0.}
\]

Un seed issu d'une seule présentation vérifie :

\[
W=\beta r\le\beta
\]

et ne peut donc pas être mûr au même pas. Sans nouvelle alimentation, `W_a` est seulement multiplié par `λ≤1` ; le temps seul ne peut jamais créer une `CELL`.

Une composante mûre devient une `CELL` portant exactement le même noyau :

\[
\boxed{H_{new}=K_a.}
\]

La promotion ne crée aucun payload cognitif supplémentaire et n'a aucun coût matériel marginal.

### 4.4 Normalisation de frontière

Après les mises à jour de `CELL` et de `Σ_L`, une `LAYER` est ramenée à une forme canonique unique avant toute croissance matérielle :

1. supprimer tout noyau de centre `C=0` ;
2. coalescer, séparément dans les `CELL` et dans `Σ_L`, les noyaux de géométrie exactement identique `(C,V)` par somme de support ;
3. promouvoir simultanément toutes les composantes mûres issues exclusivement des noyaux de `Σ_L` préexistants au snapshot ;
4. coalescer à nouveau les `CELL` de géométrie exactement identique ;
5. supprimer de `Σ_L` tout noyau qui, considéré comme présentation `(1,C_a,V_a)`, est concerné par au moins une `CELL` courante ;
6. annuler toute demande de seed qui, considérée comme présentation `(1,C_s,V_s)`, est concernée par au moins une `CELL` courante.

La forme normalisée vérifie donc : aucun centre nul, aucun clone exact dans une même population, aucune composante privée déjà couverte par une `CELL`.

Les seeds survivants restent des demandes de croissance transitoires et ne deviennent persistants qu'au §7.3. Leur admissibilité persistante n'est définitive qu'après projection dans le format scalaire de l'état. Toute `CELL`, tout seed ou toute `LAYER` créé pendant le pas n'acquiert d'autorité perceptive qu'à la présentation suivante.

---

## 5. Présentation reconnue, contexte, sortie et récursion

### 5.1 Valeurs reconnues et poids de connaissance

Pour chaque atome présenté :

\[
X_s=(r_s,c_s,v_s),
\]

définir l'ensemble exact des valeurs reconnues :

\[
\boxed{
R_s=\{C_i^-:\ i\in I_s\}/=,
}
\]

où `/=` quotient les centres vectoriellement exactement identiques.

Pour chaque `C\in R_s`, définir le gain géométrique déjà déterminé par `CONCERN` :

\[
\boxed{
g_C(X_s)
=
D_0(X_s)-D_C(X_s)
=
\|c_s\|^2-\|c_s-C\|^2
>0.
}
\]

Deux `CELL` concernées de même centre définissent la même valeur reconnue et le même gain. Elles sont quotientées avant toute pondération.

Si `R_s=\varnothing`, l'atome ne produit aucune connaissance reconnue.

Si `R_s\ne\varnothing`, chaque valeur `C\in R_s` reçoit la masse cognitive :

\[
\boxed{
\omega_{sC}
=
r_s
\frac{g_C(X_s)}{\sum_{D\in R_s}g_D(X_s)}.
}
\]

Ainsi :

\[
\boxed{
\omega_{sC}>0,
\qquad
\sum_{C\in R_s}\omega_{sC}=r_s.
}
\]

Cette pondération n'utilise ni support `A_i`, ni responsabilité `\rho_{is}`, ni dispersion `V_i`, ni identité, ni âge, ni ordre. `ALLOCATE` conserve exclusivement son rôle d'autorité d'apprentissage.

### 5.2 Présentation reconnue d'une LAYER et contexte

La présentation reconnue éphémère de la `LAYER` est :

\[
\boxed{
\mathcal K_L
=
\biguplus_{s:R_s\ne\varnothing}
\biguplus_{C\in R_s}
(\omega_{sC},C,0),
}
\]

avec coalescence exacte des noyaux ponctuels de même centre par addition de masse.

`\mathcal K_L` est exactement ce que la `LAYER` sait de sa présentation courante. Sa masse vaut :

\[
\boxed{
|\mathcal K_L|
=
\sum_{s:R_s\ne\varnothing}r_s
\le|\mathcal P|.
}
\]

S'il n'existe aucune reconnaissance, `\mathcal K_L` est vide et aucun noyau de contexte n'existe.

Sinon, le noyau de contexte est exactement la fusion :

\[
\boxed{
H_L^{\uparrow}
=
\bigoplus\mathcal K_L
=
(W_L^{\uparrow},C_L^{\uparrow},V_L^{\uparrow}).
}
\]

Sa masse vérifie :

\[
\boxed{
W_L^{\uparrow}=|\mathcal K_L|.
}
\]

`C_L^{\uparrow}` et `V_L^{\uparrow}` sont donc le barycentre et la dispersion entre les valeurs reconnues, pondérées uniquement par la masse causale des atomes et les rapports de gains de `CONCERN` à l'intérieur de chaque atome.

Le noyau nul de reste du §5.5 n'appartient jamais à `\mathcal K_L` et n'est jamais fusionné dans `H_L^{\uparrow}`.

### 5.3 Autorité verticale

Une `LAYER` possède un contexte vertical émissible si et seulement si :

\[
\boxed{
H_L^{\uparrow}\text{ existe}
\quad\land\quad
V_L^{\uparrow}>0
\quad\land\quad
C_L^{\uparrow}\ne0.
}
\]

- `V_L^{\uparrow}=0` signifie que toute la reconnaissance se réduit à une seule position vectorielle distincte ; aucune relation entre connaissances distinctes n'est donc formée ;
- `C_L^{\uparrow}=0` signifie que le contexte ne possède aucune direction vectorielle canonique ; il reste silencieux.

Aucune direction arbitraire n'est construite pour sauver un contexte exactement centré en zéro.

Lorsqu'il est émissible, l'entrée de la `LAYER` suivante est exactement la présentation singleton :

\[
\boxed{
\operatorname{input}(L_{k+1})
=
\{H_{L_k}^{\uparrow}\}.
}
\]

Il n'existe donc jamais d'arbre de branches inter-couches : une `LAYER` émet au plus un noyau de contexte par présentation.

### 5.4 Limite de résolution contextuelle

Le contrat vertical conserve exactement le quotient `(W,C,V)` de la présentation reconnue `\mathcal K_L`. Deux configurations distinctes produisant exactement le même noyau contextuel sont indiscernables pour les couches supérieures.

Cette perte est native au type cognitif d'Auxein : aucune covariance, orientation de second ordre ou identité de constituant n'est transmise.

La pondération par gain ne change pas l'ensemble des positions reconnues et ne peut donc pas, en arithmétique réelle, transformer `V_L^{\uparrow}=0` en `V_L^{\uparrow}>0` ou l'inverse. Elle peut en revanche déplacer exactement le barycentre et donc changer le statut `C_L^{\uparrow}=0`, ce qui peut créer ou supprimer une autorité verticale.

Une relation parfaitement symétrique sous ses masses cognitives, telle qu'un contexte constitué de `+a` et `-a` à masses reconnues égales, n'a aucun représentant vectoriel non nul compatible avec l'invariance orthogonale. Elle reste silencieuse.

### 5.5 Sortie géométrique d'une LAYER et famille présente

Pour toute `LAYER k` dont `\mathcal K_{k,t}` est non vide, définir :

\[
\boxed{
\mathcal Y^G_{k,t}
=
\mathcal K_{k,t}
\uplus
\{(m^0_{k,t},0,0)\},
}
\]

avec :

\[
\boxed{m^0_{k,t}=1-|\mathcal K_{k,t}|.}
\]

Le noyau nul n'est matérialisé que si `m^0_{k,t}>0`. Après coalescence :

\[
\boxed{|\mathcal Y^G_{k,t}|=1.}
\]

Le centre émis pour une valeur reconnue est exactement le centre de snapshot `C_i^-` et sa dispersion extérieure vaut `0`. La dispersion persistante `V_i` de la `CELL` décrit son histoire d'admissibilité ; elle n'est pas la valeur reconnue et n'est jamais émise.

Le noyau `(w,0,0)` est un opérateur de reste : il porte une masse sans direction cognitive. Il ne concerne aucune `CELL`, n'alimente ni `CELL` ni `Σ`, ne produit aucun seed et ne participe ni à `\mathcal K_L` ni à `H_L^{\uparrow}`.

La famille géométrique présente du `NETWORK` est :

\[
\boxed{
\mathfrak Y_t^G
=
(\mathcal Y^G_{k,t})_{k\in K_t},
}
\]

où `K_t` est l'ensemble ordonné des `LAYER` parcourues dont `\mathcal K_{k,t}` est non vide.

Chaque membre de cette famille possède son propre univers de masse. Les poids de deux `LAYER` différentes ne sont pas comparables et ne sont jamais renormalisés ensemble. L'ordre de `K_t` conserve seulement la profondeur d'origine ; il n'a aucune signification temporelle.

La famille vide est un résultat réel d'une présentation traitée : absence de connaissance ne signifie pas absence de frontière causale.

### 5.6 Récursion du NETWORK

Le `NETWORK` est une suite ordonnée :

```text
L0 → L1 → L2 → ...
```

`L0` reçoit la présentation extérieure canonique du §1.1.

Pour chaque `LAYER` suivante qui existait déjà au début de la présentation, elle reçoit l'unique noyau de contexte émissible produit par la couche précédente. Si aucun contexte émissible n'est produit, aucune couche supérieure n'est parcourue ; il n'existe qu'une branche.

Une `LAYER` sans `CELL` ne produit aucune présentation reconnue ni contexte vertical. Elle apprend uniquement les noyaux reçus dans `Σ_L`.

### 5.7 Croissance verticale

Si une `LAYER` terminale produit un contexte émissible et qu'aucune `LAYER` suivante n'existe, la géométrie demande la création d'une nouvelle `LAYER` vide, seulement si `β>0`.

Cette création appartient à la transaction globale du §7.3. Si elle est refusée, l'état cognitif existant reste inchangé. Le contexte courant n'est rejoué ni mémorisé hors de toute `LAYER`.

Une `LAYER` créée pendant la présentation ne lit pas le contexte qui a provoqué sa création. Une nouvelle profondeur exige donc au moins une nouvelle occurrence future du contexte.

### 5.8 Espace temporel privé du mode predictive

En mode `predictive`, pour chaque `LAYER L_k` de monde `E=\mathbb R^D`, le `NETWORK` possède deux populations finies dans :

\[
\boxed{T(E)=E\oplus E\simeq\mathbb R^{2D}.}
\]

Elles sont `\Sigma_k^T` et une population de `CELL` temporelles.

Ces populations utilisent sans modification les lois des §2, §3 et §4, avec dimension `2D`. Elles sont associées à `L_k` mais n'appartiennent pas à sa cognition géométrique : `L_k` ne les lit jamais.

Les deux espaces sont strictement étanches :

- une `CELL` géométrique ne concerne jamais une présentation temporelle ;
- une `CELL` temporelle ne concerne jamais une présentation géométrique ;
- les deux populations ne partagent ni `Σ`, ni allocation, ni promotion, ni contexte cognitif ;
- aucune `CELL` temporelle ne participe au contexte vertical du §5.2 ni à la croissance verticale du §5.7 ;
- aucune `CELL` temporelle n'est exposée comme connaissance extérieure.

Leur seule coexistence est structurelle dans le même `NETWORK`, économique au §7 et prédictive au §5.12.

### 5.9 Registre précédent et présentation temporelle privée

Pour chaque `L_k`, le `NETWORK` entretient en mode `predictive` un registre causal :

\[
\boxed{P_k\in\{\varnothing\}\cup\{(W,C,V):W>0,\ C\in E,\ V\ge0\}.}
\]

`P_k` contient exactement le noyau `H_{L_k}^{\uparrow}` produit par la présentation précédente **de la même séquence extérieure**, lorsqu'il existait. Il n'existe aucun historique au-delà de ce registre unique.

À l'ouverture de toute séquence :

\[
\boxed{P_k:=\varnothing\quad\forall k.}
\]

Soient deux présentations strictement adjacentes à l'intérieur de cette séquence :

\[
P_k=(W_-,C_-,V_-),
\qquad
H_{L_k,t}^{\uparrow}=(W_+,C_+,V_+).
\]

Si les deux noyaux existent, le `NETWORK` construit l'unique atome temporel :

\[
\boxed{
X_{k,t}^T
=
\left(
W_-W_+,
C_-\oplus C_+,
V_-+V_+
\right).
}
\]

Il s'agit exactement du noyau centré de la mesure produit des deux contextes dans l'espace somme directe. En particulier :

\[
\boxed{0<W_-W_+\le1}
\]

et :

\[
\boxed{
\|(x_-,x_+)-(C_-,C_+)\|^2
=
\|x_--C_-\|^2+\|x_+-C_+\|^2.
}
\]

Aucune covariance entre les deux extrémités n'est requise.

La présentation temporelle privée est le singleton :

\[
\boxed{\mathcal P_{k,t}^T=\{X_{k,t}^T\}.}
\]

Le noyau `H_{L_k,t}^{\uparrow}` est utilisable ici dès qu'il existe, indépendamment de l'autorité verticale du §5.3. Un singleton reconnu (`V=0`) ou un contexte de centre nul reste donc un état temporel valide.

Si `P_k` ou `H_{L_k,t}^{\uparrow}` est absent, aucune présentation temporelle n'existe pour `L_k` à cette présentation. L'absence ne vaut jamais présentation nulle.

Après le traitement temporel :

\[
\boxed{
P_k\leftarrow
\begin{cases}
H_{L_k,t}^{\uparrow},&\text{si ce noyau existe},\\
\varnothing,&\text{sinon}.
\end{cases}
}
\]

Cette mise à jour s'applique à toutes les `LAYER` existantes au début de la présentation. Une `LAYER` non parcourue ou ne produisant aucune reconnaissance obtient donc `P_k=\varnothing`; aucune transition sautant une présentation sans contexte ne peut être fabriquée.

Le registre avance également à `eta=0`. Une `LAYER` créée pendant la présentation commence avec `P_k=\varnothing`.

À la fermeture de la séquence :

\[
\boxed{P_k:=\varnothing\quad\forall k.}
\]

Aucune succession ne traverse une frontière de séquence. Une séquence atomique ouvre et ferme donc avec tous ses registres vides et ne peut apprendre aucune transition.

### 5.10 Cognition temporelle privée

Lorsqu'une présentation `\mathcal P_{k,t}^T` existe, les `CELL` temporelles préexistantes et `\Sigma_k^T` appliquent exactement :

```text
CONCERN
→ ALLOCATE
→ apprentissage des CELL temporelles
→ inconnu vers Σᵀ
→ DETECT
→ promotion
→ normalisation
```

avec les mêmes lois et les mêmes frontières de snapshot que dans `E`.

Une `CELL` temporelle possède donc exactement un noyau :

\[
\boxed{H_j^T=(A_j^T,C_j^T,V_j^T),\qquad C_j^T\in E\oplus E.}
\]

Écrire, uniquement comme projections géométriques :

\[
C_j^T=C_{j,-}^T\oplus C_{j,+}^T.
\]

`C_{j,-}^T` et `C_{j,+}^T` ne sont ni des identités, ni des pointeurs, ni des références vers des `CELL` géométriques.

Une connaissance temporelle représente exclusivement une succession adjacente privée. Elle ne produit aucun readout de séquence et aucun noyau destiné à un autre espace temporel :

\[
\boxed{T(T(E))\text{ n'appartient pas à Auxein}.}
\]

### 5.11 Limite de résolution temporelle

Le contrat temporel privé conserve exactement le quotient :

\[
\boxed{
(W_-W_+,\ C_-\oplus C_+,\ V_-+V_+).
}
\]

Il ne conserve pas séparément `V_-` et `V_+`. Deux transitions produisant exactement le même noyau temporel sont cognitivement indistinguables.

`CONCERN` s'applique au couple complet dans `E\oplus E`. Une meilleure correspondance sur une extrémité peut donc compenser une moins bonne correspondance sur l'autre dans la distance quadratique totale. Il n'existe aucun test canonique séparé `CONCERN(source) ∧ CONCERN(target)`.

Si `C_-=0` et `C_+=0`, le centre temporel vaut exactement zéro. Conformément aux §2.4 et §4.2, cette présentation peut faire avancer l'horloge temporelle mais ne peut être reconnue, alimenter `Σ^T` ni créer une connaissance. Aucune direction artificielle n'est construite pour distinguer `0→0`.

### 5.12 Projection prédictive et présentations futures candidates

Cette section s'applique uniquement en mode `predictive`. Elle ne définit aucune nouvelle mémoire ni aucune nouvelle population.

Soit le noyau de contexte géométrique courant d'une `LAYER` :

\[
H_{L_k,t}^{\uparrow}=(W_t,C_t,V_t),
\]

lorsqu'il existe, y compris s'il est verticalement silencieux.

Pour toute `CELL` temporelle **préexistante au début de la frontière temporelle de cette présentation** :

\[
H_j^T=(A_j^T,S_j\oplus T_j,V_j^T),
\]

le `NETWORK` considère uniquement les deux noyaux ponctuels :

\[
\boxed{X_t^P=(1,C_t,0),\qquad H_{j,-}^P=(1,S_j,0).}
\]

La `CELL` temporelle `j` est prédictivement concernée si et seulement si le `CONCERN` canonique du §2.4 est vrai :

\[
\boxed{
\|C_t-S_j\|^2<\|C_t\|^2
\quad\land\quad
\|C_t-S_j\|^2<\|S_j\|^2.
}
\]

Ni `A_j^T`, ni `V_j^T`, ni `W_t`, ni `V_t` ne modifient l'identité des candidats. Le quotient temporel du §5.11 ne permet pas de reconstruire une dispersion source ou cible ; la projection prédictive porte canoniquement sur les centres seulement.

Pour toute relation prédictivement concernée, définir son gain ponctuel et son gain relatif :

\[
\boxed{
g_j^P
=
\|C_t\|^2-\|C_t-S_j\|^2
>0,
}
\]

\[
\boxed{
\gamma_j^P
=
\frac{g_j^P}{\|C_t\|^2}
=
1-\frac{\|C_t-S_j\|^2}{\|C_t\|^2},
\qquad
0<\gamma_j^P\le1.
}
\]

Un candidat concerné implique `C_t\ne0`; le dénominateur est donc strictement positif en arithmétique réelle. `\gamma_j^P` mesure uniquement l'autorité géométrique locale de la correspondance source. Une source exactement égale au contexte courant possède `\gamma_j^P=1`.

L'autorité locale portée par cette relation est :

\[
\boxed{w_j^P=W_t\gamma_j^P,\qquad 0<w_j^P\le W_t.}
\]

Les cibles distinctes ne sont jamais normalisées entre elles. Ajouter, retirer ou modifier une relation vers une autre cible ne modifie pas le poids d'un candidat existant. Il n'existe notamment aucune contrainte `\sum_T w_T^P=W_t`.

Pour une cible extérieure distincte `T`, poser :

\[
J_T=
\{j:\ T_j=T\text{ et }S_j\text{ est prédictivement concerné}\}.
\]

Si plusieurs relations concernées projettent exactement vers la même cible, la projection extérieure oublie leur provenance et conserve l'enveloppe idempotente de leur autorité :

\[
\boxed{
w_{k,t,T}^P
=
W_t\max_{j\in J_T}\gamma_j^P.
}
\]

Ce `max` n'est pas un WTA entre futurs : toutes les cibles distinctes restent émises. Il empêche uniquement le nombre de chemins vers une même valeur extérieure de créer artificiellement de l'autorité.

Pour chaque cible distincte `T` telle que `J_T\ne\varnothing`, produire une présentation future candidate indépendante :

\[
\boxed{
\mathcal Y^P_{k,t,T}
=
\{(w_{k,t,T}^P,T,0),(1-w_{k,t,T}^P,0,0)\},
}
\]

avec suppression des masses nulles et coalescence des noyaux de même centre. Ainsi :

\[
\boxed{|\mathcal Y^P_{k,t,T}|=1.}
\]

La cible reste exactement `T_j`; aucune dispersion cible n'est inventée. Aucun support, dispersion, âge, identité, responsabilité d'apprentissage, nombre de `CELL` ni normalisation entre cibles distinctes ne possède d'autorité prédictive. Le gain relatif de `CONCERN` pondère uniquement la masse locale de chaque candidat.

Une cible `T=0` est une prédiction explicite valide. Après coalescence de `(w,0,0)` et `(1-w,0,0)`, sa présentation vaut `{(1,0,0)}` : le poids directionnel n'est plus observable, mais la présence du candidat dans la famille future reste distincte de l'absence de prédiction.

Poser `\mathfrak Y_t^P` la famille exacte des présentations futures candidates produites par toutes les `LAYER`, avec coalescence des présentations exactement identiques.

Aucune prédiction n'est relue par Auxein, ne modifie `P_k`, ne forme une présentation temporelle et ne déclenche une prédiction de profondeur supérieure. Si `A→B` et `B→C` sont connus, un présent `A` peut émettre `B` mais jamais `C` par fermeture transitive à la même présentation.

### 5.13 Sortie du NETWORK et composition directe

Pour une présentation extérieure traitée, la sortie `geometry` est :

\[
\boxed{
\operatorname{readout}_{N,t}=\mathfrak Y_t^G.
}
\]

En mode `predictive`, la sortie est le couple typé :

\[
\boxed{
\operatorname{readout}_{N,t}
=
(\mathfrak Y_t^G,\mathfrak Y_t^P).
}
\]

Une représentation JSON-compatible canonique peut être :

```text
geometry:
{
  "present": [presentation, ...]
}

predictive:
{
  "present": [presentation, ...],
  "future":  [presentation, ...]
}

presentation:
[
  [weight, center, variance],
  ...
]
```

Il n'existe aucun champ cognitif `sequences`. Les populations temporelles restent privées.

La sortie d'une séquence extérieure est la suite ordonnée des `readout` produits pour ses présentations ; les sorties ne sont pas fusionnées entre pas.

Pour une composition directe `N_1→N_2`, seule la famille `present` de `N_1` est admissible automatiquement. Chaque `\mathcal Y^G_{k,t}` est remise à `N_2`, **dans l'ordre canonique de profondeur `K_t`**, comme une séquence atomique indépendante :

\[
\boxed{
\mathfrak Y_t^G=(\mathcal Y^G_{k,t})_k
\Longrightarrow
\{(\mathcal Y^G_{k,t})\}_k
\text{ comme séquences de longueur }1.
}
\]

Cet ordre de profondeur peut avoir une autorité d'apprentissage géométrique aval, puisque des présentations distinctes font avancer les EMA ; il n'a en revanche **aucune autorité temporelle**, chaque membre ouvrant et fermant sa propre séquence. Une famille vide établit néanmoins une frontière de composition : aucun registre précédent du downstream ne peut lui survivre.

Cette composition transmet de la géométrie, jamais une continuité temporelle implicite. Un downstream `predictive` peut utiliser sur chaque singleton des relations déjà apprises, mais il n'apprend aucune relation temporelle par le simple branchement. Pour apprendre des successions, il doit recevoir une séquence extérieure non atomique explicite.

La famille `future` n'est jamais auto-réinjectée dans cette composition.

---
## 6. Causalité d'une séquence

À toute population effectivement présentée sont associés trois états conceptuels :

\[
\boxed{
X^-\xrightarrow{\text{perception unique}}X^*
\xrightarrow{\text{normalisation}}X^+.
}
\]

Pour une `LAYER`, ces états sont ceux du compartiment géométrique. En mode `predictive`, la population temporelle privée associée possède sa propre frontière de présentation et applique exactement la même discipline de snapshot. La lecture prédictive observe le snapshot temporel préexistant et ne constitue pas une population apprenante.

Tous les `CONCERN`, `ALLOCATE`, reconnaissances, gains de connaissance, contextes, cibles EMA et décisions privées d'un espace sont calculés exclusivement depuis son snapshot `X^-` et sa présentation courante. Aucun objet créé pendant la présentation ne peut lire, concerner, apprendre, être reconnu ou émettre pour cette même présentation. Aucun replay n'existe.

Pour chaque séquence extérieure :

1. en mode `predictive`, poser tous les `P_k:=∅` avant la première présentation ;
2. pour chaque présentation de la séquence, dans son ordre causal :
   1. restaurer d'abord la solvabilité matérielle si nécessaire (§7.4) ; toute contraction forcée invalide simultanément tous les registres `P_k` ;
   2. figer la suite des `LAYER` existantes pour cette présentation et initialiser les objets éphémères `\mathcal K`, `\mathfrak Y^G` et, en mode `predictive`, `\mathfrak Y^P` ;
   3. coalescer les atomes extérieurs de géométrie exactement identique et remettre la présentation à `L0` ;
   4. pour chaque `LAYER` existante recevant une présentation géométrique non vide, dans l'ordre du réseau :
      1. figer le snapshot géométrique ;
      2. appliquer `CONCERN/ALLOCATE` aux `CELL` géométriques du snapshot ;
      3. construire `\mathcal K_L`, `H_L^{\uparrow}` et, si `\mathcal K_L` est non vide, `\mathcal Y_L^G` depuis ce même snapshot ;
      4. si `H_L^{\uparrow}` est verticalement émissible et que la `LAYER` suivante existait au début de la présentation, lui transmettre immédiatement le singleton `{H_L^{\uparrow}}` ;
      5. mettre à jour exactement une fois les `CELL` géométriques préexistantes ;
      6. appliquer `DETECT` aux seuls atomes géométriques inconnus depuis le `Σ_L` du snapshot, puis mettre à jour exactement une fois ses composantes préexistantes ;
      7. normaliser le compartiment géométrique selon le §4.4 ;
      8. si la `LAYER` est terminale, que son contexte était émissible, qu'aucun successeur n'existait et que `β>0`, former une demande de nouvelle `LAYER` ;
   5. en mode `predictive`, après la phase géométrique complète, pour chaque `LAYER` qui existait au début de la présentation :
      1. prendre le registre `P_k` et le noyau courant `H_{L_k,t}^{\uparrow}`, éventuellement absents ;
      2. figer le snapshot des `CELL` temporelles préexistantes ; si le contexte courant existe, produire les candidats de `\mathfrak Y_t^P` depuis ce snapshot selon le §5.12 ;
      3. si `P_k` et le contexte courant existent, construire `\mathcal P_{k,t}^T` selon le §5.9 ;
      4. si cette présentation existe, appliquer `CONCERN/ALLOCATE` depuis le même snapshot temporel, mettre à jour exactement une fois les `CELL` temporelles et `Σ_k^T`, puis normaliser ce compartiment selon les §3 et §4 ;
      5. remplacer `P_k` par le noyau courant ou par `∅` selon le §5.9 ;
   6. réunir tous les seeds géométriques survivants, en mode `predictive` tous les seeds temporels survivants, et l'éventuelle demande de `LAYER` en une transaction globale de croissance (§7.3) ;
   7. exécuter cette transaction entière si elle est payable, sinon ne rien créer ;
   8. pour toute nouvelle `LAYER` admise en mode `predictive`, créer son compartiment temporel vide et son registre `P_k=∅` sans lui faire lire la présentation courante ni prédire ;
   9. matérialiser le `readout` de la présentation selon le §5.13 ;
3. en mode `predictive`, poser tous les `P_k:=∅` après la dernière présentation de la séquence.

La phase temporelle privée ne peut modifier aucune reconnaissance géométrique de la même présentation. La phase géométrique ne lit aucun état temporel. Leur seul raccord causal apprenant est la construction par le `NETWORK` de `\mathcal P_{k,t}^T` depuis deux contextes géométriques adjacents de la même séquence. La projection prédictive est une lecture supplémentaire unidirectionnelle du snapshot temporel vers la famille future seulement.

Une séquence atomique exécute exactement la même phase géométrique et la même lecture prédictive, mais ses resets d'ouverture et de fermeture empêchent toute présentation temporelle issue d'une autre séquence ou destinée à la suivante.

---
## 7. Économie matérielle

### 7.1 Principe

Le budget est un plafond entier exact :

\[
\boxed{B_{units}\in\mathbb N.}
\]

L'empreinte persistante de l'état est :

\[
\boxed{M_{units}(\mathcal A)\in\mathbb N.}
\]

Un état est solvable si et seulement si :

\[
\boxed{M_{units}(\mathcal A)\le B_{units}.}
\]

Le budget n'est pas une monnaie accumulée. Une destruction cesse d'occuper de la capacité ; elle ne crée aucun crédit.

L'économie ne modifie aucune loi géométrique.

### 7.2 Packing canonique

Soit :

- `p=4` pour un réel persistant `f32` ;
- `p=8` pour un réel persistant `f64` ;
- un `u64` coûte `8` unités ;
- un tag discret coûte `1` unité.

Le noyau géométrique `(W,C,V)` en dimension `D` occupe :

\[
\boxed{U_H=(D+2)p.}
\]

Le noyau temporel en dimension `2D` occupe :

\[
\boxed{U_T=(2D+2)p.}
\]

Une composante de `Σ_L` et une `CELL` géométrique occupent `U_H`. Une composante de `Σ_k^T` et une `CELL` temporelle occupent `U_T`.

Le header logique du `NETWORK` contient :

- `format_version=5`, `dimension`, `steps_seen`, `layer_count` sur `u64` ;
- un tag `scalar` ;
- un tag `mode` ;
- `memory`, `eta` dans le format persistant.

Donc :

\[
\boxed{U_N=34+2p.}
\]

En mode `geometry`, chaque `LAYER` possède deux compteurs `u64` :

\[
\boxed{U_L^G=16.}
\]

En mode `predictive`, chaque `LAYER` possède :

- quatre compteurs `u64` pour `Σ_L`, `CELL`, `Σ_k^T`, `CELL^T` ;
- un tag de présence du registre `P_k` ;
- un slot fixe de noyau géométrique `U_H` réservé à `P_k`, qu'il soit présent ou absent.

Ainsi :

\[
\boxed{U_L^P=33+U_H.}
\]

La réservation fixe de `P_k` garantit que l'observation du contexte précédent à l'intérieur d'une séquence explicite ne constitue jamais une croissance matérielle hors transaction. Le contenu absent du slot n'a aucune autorité cognitive ; une frontière de séquence le remet à `∅` sans modifier l'empreinte.

Pour `N_C(L)` `CELL` géométriques et `N_T(L)` `CELL` temporelles :

En mode `geometry` :

\[
\boxed{
M_{units}(\mathcal A)
=
U_N+
\sum_L
\left[U_L^G+(|\Sigma_L|+N_C(L))U_H\right].
}
\]

En mode `predictive` :

\[
\boxed{
M_{units}(\mathcal A)
=
U_N+
\sum_L
\left[
U_L^P
+(|\Sigma_L|+N_C(L))U_H
+(|\Sigma_L^T|+N_T(L))U_T
\right].
}
\]

La présentation reconnue, le contexte vertical, les présentations temporelles privées, les sorties présentes et les candidats futurs sont éphémères et ne possèdent aucun coût persistant propre. La temporalité privée du mode `predictive` est entièrement comptée dans cette empreinte ; la lecture prédictive elle-même reste éphémère.

La promotion `Σ→CELL`, dans l'un ou l'autre espace, conserve exactement le même payload. Son coût marginal est donc :

\[
\boxed{c_{promote}=0.}
\]

Chaque nouveau noyau effectivement ajouté à `Σ_L` coûte :

\[
\boxed{c_{seed}^G=U_H,}
\]

et chaque nouveau noyau ajouté à `Σ_k^T` :

\[
\boxed{c_{seed}^T=U_T.}
\]

Une nouvelle `LAYER` vide coûte :

\[
\boxed{
c_{layer}=
\begin{cases}
U_L^G,&mode=\texttt{geometry},\\
U_L^P,&mode=\texttt{predictive}.
\end{cases}
}
\]

L'état minimal exécutable est `NETWORK + L0` vide :

\[
\boxed{
M_{min}=
\begin{cases}
U_N+U_L^G,&mode=\texttt{geometry},\\
U_N+U_L^P,&mode=\texttt{predictive}.
\end{cases}
}
\]

Si `B_{units}<M_{min}`, l'environnement est inexécutable.

Une interface peut exprimer ergonomiquement le budget en unités de noyau, mais toute décision interne utilise exclusivement `B_{units}`.

### 7.3 Croissance

Les promotions du §4.4 sont géométriques et matériellement neutres ; elles sont appliquées avant toute création matérielle.

Après lecture géométrique et, en mode `predictive`, temporelle de la présentation, réunir :

- toutes les demandes de nouveaux noyaux `Σ_L` encore admissibles du §4.4 ;
- en mode `predictive`, toutes les demandes de nouveaux noyaux `Σ_k^T` encore admissibles ;
- l'éventuelle nouvelle `LAYER` de frontière requise par le §5.7.

Notons `Π_p` la projection atomique vers le format scalaire persistant du §8.1. Pour chaque demande de seed `H_s`, former :

\[
\boxed{\widehat H_s=\Pi_p(H_s).}
\]

Avant toute décision matérielle, normaliser ces demandes projetées dans leur espace propre :

1. supprimer tout `\widehat H_s` de centre nul ;
2. supprimer tout `\widehat H_s` qui, considéré comme présentation `(1,\widehat C_s,\widehat V_s)`, est concerné par au moins une `CELL` courante ;
3. coalescer exactement les noyaux projetés de même géométrie `(C,V)`, y compris avec un noyau privé persistant déjà identique.

Les créations persistantes nettes ainsi obtenues, avec l'éventuelle nouvelle `LAYER`, forment l'unique transaction de croissance matérielle `G_t`.

À `β=0`, `G_t` est vide.

`\mathcal A\oplus G_t` désigne l'état obtenu en appliquant simultanément toutes les créations de ce lot après cette normalisation persistante.

La transaction est exécutée si et seulement si :

\[
\boxed{M_{units}(\mathcal A\oplus G_t)\le B_{units}.}
\]

Sinon aucune création de `G_t` n'a lieu.

L'économie ne sélectionne jamais un sous-ensemble des demandes persistantes de `G_t`.

### 7.4 Solvabilité forcée

Une baisse de budget peut rendre l'état courant insolvable. La contraction a lieu avant toute nouvelle perception.

Si :

\[
M_{units}(\mathcal A)>B_{units},
\]

alors :

1. vider simultanément toutes les mémoires `Σ_L` et, en mode `predictive`, toutes les mémoires `Σ_k^T` ;
2. supprimer toute `LAYER` terminale sans aucune `CELL` géométrique ni temporelle, sans jamais supprimer `L0` ;
3. si l'état reste insolvable, considérer ensemble les valeurs distinctes `K_i` de toutes les `CELL` géométriques et temporelles restantes et, pour chaque valeur `k`, l'état `\mathcal A_{>k}` obtenu en conservant exactement les `CELL` des deux espaces telles que `K_i>k`, puis en supprimant les `LAYER` terminales devenues sans connaissance ;
4. s'il existe un `k` tel que `M_{units}(\mathcal A_{>k})\le B_{units}`, choisir le plus petit et remplacer l'état par `\mathcal A_{>k}` ;
5. sinon, si l'état minimal `NETWORK + L0` vide est solvable, supprimer toutes les `CELL` des deux espaces et ramener le réseau à cet état minimal ; sinon l'environnement est inexécutable ;
6. en mode `predictive`, si au moins une contraction a été nécessaire, poser simultanément `P_k=\varnothing` pour toutes les `LAYER` survivantes.

Cette dernière invalidation est causalement nécessaire : un registre précédent peut contenir la géométrie d'une connaissance que la contraction vient de détruire. Aucune succession ne traverse une frontière de destruction matérielle. Le slot réservé à `P_k` reste alloué et cette invalidation ne modifie pas l'empreinte.

La contraction détruit exactement des classes entières de même valeur, indépendamment de leur espace :

\[
\boxed{K_i=K_j\Longrightarrow(i\text{ survit}\iff j\text{ survit}).}
\]

Le support EMA, l'âge et l'absence du flux ne participent jamais à la décision. Une destruction ne réinjecte aucun passé dans `Σ_L` ni dans `Σ_k^T`.

### 7.5 Absence de remplacement volontaire

Un état solvable ne détruit aucune `CELL` afin d'en financer une autre. Si la transaction de croissance n'est pas payable, elle attend une frontière future.

`K_i` n'est consulté que lorsqu'une perte de connaissance est déjà matériellement obligatoire.

### 7.6 Mutation des paramètres

Le budget peut changer entre deux présentations. Une hausse ne crée rien immédiatement. Une baisse est résolue par le §7.4 à la frontière suivante.

Une modification de `eta` est atomique et redéfinit seulement `β` et `λ` à la frontière suivante. Elle ne crée, ne fusionne ni ne détruit aucun noyau au moment de la mutation.

`mode` est immuable pour un état existant et ne possède aucune loi de mutation.

### 7.7 Invariant et terminaison

À toute frontière solvable :

\[
\boxed{M_{units}(\mathcal A)\le B_{units}.}
\]

La contraction forcée termine car elle opère sur des populations finies, puis choisit au plus un cutoff dans l'ensemble fini des valeurs `K_i`.

Après perception, les EMA, promotions, suppressions de travail couvert, coalescences, resets de frontière et mises à jour des slots `P_k` n'augmentent pas l'empreinte ; la seule croissance persistante est `G_t`, soumise à un unique test de payabilité.

Toute transition finie termine donc sur un état solvable ou sur le verdict « environnement inexécutable ».

---

## 8. Invariances, dégénérescences et exigences numériques

Toute réalisation conforme préserve :

1. permutation des atomes d'une présentation ;
2. coalescence ou découpage d'atomes de géométrie exactement identique ;
3. rotation orthogonale de l'espace vectoriel ;
4. changement d'échelle uniforme avec `C→aC` et `V→a²V` ;
5. zero-padding exact ;
6. renommage bijectif des éventuelles poignées administratives ;
7. conservation de la masse des responsabilités d'apprentissage ;
8. pour chaque atome reconnu, `Σ_C ω_{sC}=r_s` ;
9. conservation de la masse contextuelle `W_L^{\uparrow}=Σ_{s:R_s≠∅}r_s` ;
10. indépendance de `\mathcal K_L` et du contexte vertical aux supports `A_i`, responsabilités `ρ_i`, dispersions `V_i`, identités, âges et ordres des `CELL`, une fois les valeurs reconnues déterminées ;
11. dépendance des poids relatifs de connaissances uniquement aux rapports des gains géométriques de `CONCERN` ;
12. indépendance cognitive des `LAYER` hors présentation `(r,c,v)` ;
13. absence de replay ;
14. absence de subdivision causale d'une présentation par une optimisation d'exécution ;
15. unicité du noyau contextuel émis par couche et par présentation ;
16. silence vertical d'une reconnaissance réduite à une seule position distincte ;
17. silence vertical d'un contexte exactement centré en zéro ;
18. neutralité cognitive de tout noyau `(w,0,0)` ;
19. masse exactement `1` de chaque sortie géométrique de `LAYER` et de chaque présentation future candidate ;
20. absence d'autorité de la dispersion persistante d'une `CELL` dans sa valeur extérieure reconnue ;
21. en mode `predictive`, distinction de l'ordre temporel : en général `A→B` et `B→A` occupent des centres distincts dans `E⊕E` ;
22. absence de transition à travers une présentation sans contexte reconnu dans la `LAYER` considérée ;
23. absence de transition à travers une frontière de séquence ;
24. une séquence atomique n'apprend aucune transition mais peut utiliser une relation temporelle préexistante pour prédire ;
25. indépendance cognitive stricte des populations géométriques et temporelles ;
26. absence de récursion `T(T(E))` et absence de readout temporel ;
27. même changement d'échelle uniforme et même rotation orthogonale appliqués aux deux extrémités temporelles ;
28. même économie de contraction pour des `CELL` de valeur `K` égale, quel que soit leur espace ;
29. la projection prédictive commute avec tout changement d'échelle uniforme non nul et toute rotation orthogonale appliquée simultanément au présent, à la source et au successeur ; le gain relatif `\gamma^P` est conservé ;
30. une dispersion temporelle différente à centre temporel identique ne modifie pas la lecture prédictive ;
31. plusieurs futurs distincts restent plusieurs présentations indépendantes sans normalisation mutuelle ; ajouter ou retirer une branche distincte ne modifie pas les masses des autres branches ;
32. plusieurs relations concernées vers une même cible se quotiennent par l'enveloppe `max`, idempotente sous duplication et conservative vis-à-vis de la meilleure relation ;
33. les supports, dispersions, âges, identités et nombres de `CELL` temporelles ne possèdent aucune autorité dans la masse prédictive extérieure ;
34. la composition directe de sorties présentes ne crée aucune causalité entre `LAYER`, entre familles successives ou à travers une famille vide.

L'origine `0` est sémantique ; une translation uniforme n'est donc pas une invariance exigée.

Une égalité géométrique exacte ne peut être résolue par un ID, une adresse mémoire, un ordre de conteneur ou un axe arbitraire.

Une masse nulle n'apprend rien. Une `CELL` avec `C_i=0` ne concerne aucun atome. Un contexte avec `C^{\uparrow}=0` n'est pas émis verticalement.

Deux contextes distincts ayant exactement le même quotient `(W,C,V)` sont cognitivement indistinguables pour la couche suivante. Auxein n'invente aucune structure supplémentaire pour les séparer.

Deux successions distinctes ayant exactement le même quotient `(W_-W_+,C_-⊕C_+,V_-+V_+)` sont cognitivement indistinguables pour l'espace temporel privé. La dispersion temporelle n'est pas décomposable en dispersions d'extrémité.

Une succession dont les deux centres valent exactement zéro possède un centre temporel nul et reste silencieuse. Une source temporelle projetée exactement nulle reste également silencieuse, tandis qu'une cible projetée nulle peut être émise explicitement comme candidat futur entièrement nul.

### 8.1 Calcul numérique

Les valeurs persistantes utilisent `f32` ou `f64`. Les calculs intermédiaires doivent être réalisés au moins en `binary64` avant projection atomique dans le format persistant.

Toute décision qui porte sur la validité de l'état persistant s'applique à la valeur effectivement projetée. En particulier, une demande de seed est projetée, renormalisée et retestée contre les `CELL` au §7.3 avant d'entrer dans `Σ`.

Les réductions dont l'ordre n'a aucune autorité doivent être reproductibles et indépendantes de l'ordre d'itération.

Pour la pondération du §5.1, la loi canonique reste :

\[
\omega_{sC}=r_s\frac{g_C}{\sum_Dg_D}.
\]

Une implémentation doit normaliser les gains dans des unités communes avant sommation lorsque cela est nécessaire pour éviter sous-flux ou débordement, par exemple :

\[
\hat g_C=\frac{g_C}{\max_Dg_D},
\qquad
\omega_{sC}=r_s\frac{\hat g_C}{\sum_D\hat g_D}.
\]

Cette transformation est mathématiquement identique et ne possède aucune autorité cognitive.

Les variances et fusions utilisent les formes positives du §2 ; aucune soustraction de grands moments presque égaux n'est nécessaire à la loi canonique.

Aucune valeur seulement petite ne peut être remplacée par zéro au moyen d'un epsilon comportemental. Les zéros structurellement démontrés peuvent être construits exactement.

Le test `V_L^{\uparrow}=0` signifie que toutes les valeurs reconnues distinctes fusionnées ont exactement la même position après quotient ; aucune tolérance ne crée ni ne détruit une relation verticale.

Le reste extérieur `1-|\mathcal K_L|` est mathématiquement non négatif. Une réalisation flottante doit utiliser une sommation stable et empêcher qu'une erreur de quelques ulp transforme un reste mathématiquement nul en masse négative ; cette correction numérique ne peut modifier une masse non nulle canonique.

Pour la projection prédictive du §5.12, le test ponctuel et `\gamma^P` doivent utiliser une même normalisation homogène lorsque `\|C_t\|^2`, `\|S_j\|^2` ou la distance quadratique sortent de la plage représentable directe ou entrent en régime subnormal. Dans ces unités communes, `\gamma^P=(\|C_t\|^2-\|C_t-S_j\|^2)/\|C_t\|^2` est inchangé. Aucun epsilon comportemental ne peut créer, supprimer ou renforcer un candidat.

Une implémentation doit empêcher qu'un support positif soit interprété comme une destruction cognitive uniquement à cause d'un sous-flux numérique lors de l'oubli. Une renormalisation commune ou toute représentation mathématiquement équivalente est admissible si elle conserve exactement les décisions canoniques.

### 8.2 Frontière d'implémentation

Caches, index, décroissances différées, queues, parallélisme, chunking, mémoïsation et structures de travail sont autorisés s'ils sont entièrement reconstructibles depuis l'état canonique et n'altèrent aucune décision.

Un index géométrique peut réduire les candidats aux concernements publics et privés, mais seuls les prédicats du §2.4 possèdent l'autorité.

La construction de `\mathcal K_L` et du contexte peut être incrémentale, mais elle doit être exactement équivalente à la pondération par gain du §5.1, à la coalescence des contributions ponctuelles, puis à leur fusion commutative du §2.1.

Une décroissance différée doit distinguer l'horloge de chaque espace effectivement présenté. En mode `predictive`, une présentation géométrique sans présentation temporelle privée ne fait pas vieillir les mémoires temporelles associées. Une lecture prédictive seule ne fait jamais avancer leur horloge.

Une présentation reste un événement causal unique et une frontière de séquence reste une frontière causale, quelle que soit leur réalisation physique.

---
## 9. État persistant canonique

L'état persistant est minimal.

### 9.1 NETWORK

- ordre des `LAYER` ;
- `format_version=5` administratif ;
- `dimension` ;
- `scalar∈{f32,f64}` ;
- `memory` ;
- `eta` ;
- `mode∈{geometry,predictive}` ;
- compteur de présentations achevées ;
- en mode `predictive`, un slot de registre `P_k` présent ou absent pour chaque `LAYER`.

Le budget appartient à l'environnement matériel et n'est pas une connaissance apprise.

Les registres `P_k` sont un état causal du `NETWORK`, pas une connaissance. Leur autorité est strictement bornée par une séquence extérieure explicite. Ils peuvent être persistés entre deux présentations d'une même séquence explicitement poursuivie ; une frontière de séquence les remet tous à `∅`. Sauvegarder puis recharger ne crée jamais, à lui seul, une continuité causale.

### 9.2 LAYER

Toujours :

- `Σ_L`, population finie de noyaux `(W,C,V)` en dimension `D` ;
- population de `CELL` géométriques.

En mode `predictive`, le `NETWORK` entretient en outre, structurellement associé à cette `LAYER` :

- `Σ_k^T`, population finie de noyaux `(W,C,V)` en dimension `2D` ;
- population de `CELL` temporelles.

Ces populations temporelles ne constituent pas un niveau architectural supplémentaire et ne sont jamais lues par la `LAYER`.

À toute frontière causale, chaque espace est sous la forme normalisée du §4.4 : aucun centre nul, aucun clone exact dans une même population et aucun noyau privé déjà couvert par une `CELL` de son propre espace.

Index, horloges d'exécution paresseuses et tables de travail sont dérivés et ne possèdent aucune autorité cognitive.

### 9.3 CELL

Une `CELL`, géométrique ou temporelle, possède exactement :

\[
\boxed{H_i=(A_i,C_i,V_i).}
\]

Seule la dimension de `C_i` distingue son espace : `D` pour une `CELL` géométrique, `2D` pour une `CELL` temporelle. Aucun tag cognitif `kind` n'est nécessaire.

Aucune autre mémoire cognitive n'est requise.

### 9.4 Éléments non persistants

Ne sont notamment pas persistés :

- présentations géométriques ou temporelles courantes ;
- ensembles `R_s`, gains `g_C`, poids `ω_{sC}` et présentations reconnues `\mathcal K_L` ;
- responsabilités ;
- noyaux de contexte `H_L^{\uparrow}` de la présentation courante après transfert éventuel dans `P_k` ;
- sorties géométriques `\mathcal Y^G`, familles présentes `\mathfrak Y^G` et familles futures `\mathfrak Y^P` ;
- demandes de croissance non encore commises ;
- caches et index d'exécution.

---
## 10. Fermeture

Pour un état canonique fini `\mathcal A`, une séquence extérieure finie non vide :

\[
\mathcal S=(\mathcal P_0,\ldots,\mathcal P_{m-1}),
\]

une configuration valide et un environnement matériel exécutable, les sections précédentes définissent une transition finie et une suite de sorties éphémères :

\[
\boxed{
(\mathcal A,\mathcal S)
\longmapsto
(\mathcal A',\operatorname{readout}_0,\ldots,\operatorname{readout}_{m-1}).
}
\]

En mode `predictive`, tous les registres `P_k` valent `∅` à la frontière d'entrée et à la frontière de sortie de cette séquence. La causalité interne est exclusivement :

\[
\boxed{
\mathcal P_{t-1}\to\mathcal P_t
\quad\text{pour deux présentations adjacentes de la même séquence.}
}
\]

Le noyau cognitif géométrique d'une `LAYER` est :

```text
présentation de noyaux
→ CELL concernées
→ valeurs reconnues pondérées par leur gain de CONCERN
→ présentation reconnue K
→ contexte reconnu unique
→ LAYER suivante
```

Ce qu'aucune `CELL` géométrique ne reconnaît suit :

```text
inconnu
→ Σ_L
→ récurrence
→ CELL géométrique locale
```

En mode `predictive`, le `NETWORK` ajoute privément :

```text
contexte reconnu précédent dans la même séquence
+ contexte reconnu courant
→ noyau produit dans E⊕E
→ CELL temporelles concernées
→ inconnu vers Σᵀ
→ récurrence
→ CELL temporelle privée
```

et lit, sans apprentissage :

```text
contexte reconnu courant
+ projection source d'une CELL temporelle préexistante
→ CONCERN ponctuel
→ gain relatif de la source
→ projection successeur
→ enveloppe max pour une même cible
→ présentation future candidate pondérée
```

La croissance horizontale, la croissance verticale, l'apprentissage temporel privé et la lecture prédictive sont donc distincts :

\[
\boxed{
\text{inconnu récurrent}
\longrightarrow
\text{nouvelle connaissance dans le même espace},
}
\]

\[
\boxed{
\text{connaissances reconnues pondérées dans une même présentation}
\longrightarrow
\text{contexte de la LAYER suivante},
}
\]

\[
\boxed{
H_{k,t-1}^{\uparrow},H_{k,t}^{\uparrow}
\longrightarrow
\text{présentation temporelle privée dans }T(E_k),
}
\]

\[
\boxed{
C_{k,t}^{\uparrow},\ S_j\oplus T_j
\longrightarrow
\{(W_{k,t}^{\uparrow},T_j,0),(1-W_{k,t}^{\uparrow},0,0)\}
\quad\text{si }S_j\text{ concerne ponctuellement }C_{k,t}^{\uparrow}.
}
\]

Les quatre primitives cognitives nommées restent :

```text
CONCERN
ALLOCATE
DETECT
CONTEXT
```

`CONCERN` et `ALLOCATE` utilisent l'unique primitive de population du §2.4. `DETECT` applique cette même géométrie privément à `Σ`, dans `E` comme dans `T(E)`. `CONTEXT` construit `\mathcal K_L` depuis les gains de `CONCERN`, puis la fusionne en `H_L^{\uparrow}` pour la récursion verticale. La construction du noyau temporel et la projection prédictive sont des responsabilités structurelles du `NETWORK`, pas de nouvelles lois cognitives.

Une abstraction supérieure est une régularité récurrente du contexte compact des connaissances déjà reconnues par l'étage précédent. Une connaissance temporelle est une régularité récurrente entre deux de ces contextes strictement adjacents dans une séquence explicite. Elle ne possède ni histoire plus profonde, ni pointeur vers ses extrémités conceptuelles, ni autorité verticale, ni représentation extérieure propre. Sa projection source peut seulement servir de clé géométrique ponctuelle pour exposer sa projection droite comme futur connu possible.

La sortie présente est toujours géométrique. En mode `predictive`, des présentations futures candidates lui sont adjointes dans un canal séparé. Plusieurs `LAYER` et plusieurs futurs conservent chacun leur propre univers de masse ; aucune pondération mutuelle n'est inventée.

La composition directe entre deux Auxein transmet uniquement les familles géométriques présentes comme séquences atomiques indépendantes. Elle ne transmet aucune causalité implicite et n'auto-réinjecte jamais les futurs prédits.

La géométrie détermine les connaissances présentes, les concernements et les créations admissibles. Les frontières de séquence déterminent quelles adjacences ont une autorité causale. L'économie maintient un état fini sans sélectionner arbitrairement entre créations simultanées. Une connaissance acquise persiste indépendamment de son actualité et ne peut être perdue que lorsqu'une contraction matérielle est obligatoire, selon sa valeur géométrique intrinsèque.
