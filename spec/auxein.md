# AUXEIN v0.4.0 — Canon mathématique et matériel

**Version : 0.4.0**  
**Statut : canon mathématique et matériel**

\[
\boxed{\text{la géométrie pousse ; l'économie autorise}}
\]

## 0. Contrat

Auxein est un réseau ordonné de `LAYER` autonomes. Chaque `LAYER` applique exactement la même transformation géométrique :

```text
présentation
→ concernement par les CELL
→ partage de masse
→ apprentissage local des CELL
→ inconnu vers Σ
→ contexte unique des reconnaissances
→ présentation de la LAYER suivante
```

Le `NETWORK` fonctionne dans exactement un mode causal :

```text
geometry
temporal
predictive
```

Les modes sont strictement cumulatifs :

\[
\boxed{\texttt{geometry}\subset\texttt{temporal}\subset\texttt{predictive}.}
\]

`geometry` applique seulement la transformation géométrique. `temporal` conserve cette géométrie sans modification et ajoute, pour chaque `LAYER`, un espace de succession `T(E)=E⊕E` appartenant au `NETWORK`, qui apprend les passages strictement adjacents `step-1 → step` entre contextes reconnus. `predictive` conserve intégralement les deux précédents et ajoute une lecture éphémère des `CELL` temporelles existantes : le contexte reconnu courant peut concerner leur projection source et émettre leur projection successeur comme futur connu possible.

Une `CELL` représente une connaissance directionnelle acquise. Elle se déclare concernée uniquement par sa propre géométrie. Plusieurs `CELL` peuvent se déclarer concernées simultanément.

Ce qu'aucune `CELL` ne reconnaît reste local à la `LAYER`, alimente une mémoire privée `Σ`, et peut devenir une nouvelle `CELL` lorsqu'il est récurrent.

Ce qui traverse une frontière de couche n'est pas une erreur, une provenance ni une branche par `CELL`. Une `LAYER` compresse toutes les valeurs qu'elle a effectivement reconnues pendant la présentation en **un unique noyau de contexte**. Ce noyau devient, lorsqu'il possède une diversité et une direction non nulles, l'unique présentation de la `LAYER` suivante.

Une présentation extérieure est une observation logique simultanée. Une présentation multi-vecteur affirme donc que ses vecteurs appartiennent au même contexte causal ; la découper en plusieurs appels successifs constitue plusieurs présentations différentes.

Principes normatifs :

1. aucune tâche externe, cible, classe, label ou loss supervisée ;
2. aucune `LAYER` ne lit l'état privé d'une autre ;
3. le seul contrat cognitif inter-couches est une présentation finie positive de noyaux centrés ;
4. toutes les `LAYER` appliquent exactement la même loi ;
5. aucun WTA, `top-k`, choix cognitif par identité ou ordre d'itération ;
6. plusieurs `CELL` peuvent être concernées simultanément par le même atome ;
7. ce qui est inconnu apprend horizontalement dans la `LAYER` courante ;
8. seules les connaissances effectivement reconnues participent au contexte vertical ;
9. une couche émet au plus un noyau de contexte par présentation ;
10. aucune responsabilité d'apprentissage, masse interne de `CELL`, identité ou provenance ne pondère la géométrie du contexte vertical ;
11. aucun seuil numérique arbitraire ni epsilon comportemental ;
12. aucune autorité cognitive de l'âge, d'une provenance ou d'une identité administrative ;
13. une présentation est causalement atomique ;
14. l'ordre des atomes d'une présentation n'a aucune autorité ;
15. un objet créé pendant un pas ne lit ni n'émet pour ce pas ;
16. la géométrie cognitive est définie indépendamment du budget ;
17. toute quantité exactement reconstructible peut rester éphémère ;
18. l'origine `0` n'est pas une connaissance : elle représente l'absence de direction cognitive canonique ;
19. une relation exactement symétrique de centre nul reste silencieuse plutôt que de recevoir un axe arbitraire ;
20. aucune matrice, covariance persistante, axe privilégié ou géométrie de second ordre n'appartient à Auxein ;
21. le temps canonique est exclusivement l'ordre discret `step-1 → step` ; aucune horloge physique n'est implicite ;
22. les `CELL` géométriques et temporelles n'apprennent, ne concernent et ne se concurrencent jamais dans le même espace ;
23. le `NETWORK` est seul responsable de la construction des présentations temporelles et de leur mémoire causale précédente ;
24. les connaissances temporelles n'émettent aucun contexte vertical et ne forment aucun `T(T(E))` ;
25. géométrie et temporalité partagent une économie matérielle unique ;
26. le `readout` peut réunir les reconnaissances géométriques et temporelles d'un même pas sans créer aucun lien cognitif persistant entre elles ;
27. la prédiction ne crée, n'apprend ni ne modifie aucune connaissance : elle projette uniquement des `CELL` temporelles préexistantes ;
28. une prédiction n'est jamais réinjectée comme présentation, contexte, mémoire causale ou entrée d'une autre prédiction ;
29. plusieurs futurs concernés sont tous émis ; aucune sélection, probabilité, WTA ou classement prédictif n'existe.

---

## 1. Présentations et horloge

### 1.1 Présentation extérieure

Soit `D∈N*`. Une présentation extérieure est une liste finie non vide de vecteurs :

\[
\boxed{
\mathcal X=(x_1,\dots,x_n),
\qquad n>0,
\qquad x_s\in\mathbb R^D.
}
\]

Son ordre n'a aucune autorité. `NETWORK` lui associe à l'entrée de `L0` la présentation uniforme :

\[
\boxed{
\mathcal P_0=
\left\{\left(\frac1n,x_s,0\right)\right\}_{s=1}^{n}.
}
\]

La masse totale extérieure vaut donc exactement `1`.

### 1.2 Atome interne

Toute présentation reçue par une `LAYER` est une famille finie positive de noyaux-atome :

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
- `v_s` : dispersion scalaire interne autour de ce centre.

Une entrée extérieure `x` est donc exactement le cas dégénéré `(r,x,0)`.

Les atomes de géométrie exactement identique `(c,v)` sont coalescés par somme de masse avant tout calcul. Leur ordre et leur découpage artificiel n'ont aucune autorité.

Poser :

\[
\boxed{|\mathcal P|:=\sum_s r_s.}
\]

Une présentation est une unité de contexte causal, pas un batch d'exécution. Regrouper ou séparer deux observations non simultanées appartient à l'application hôte et peut modifier la cognition.

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

Les connaissances et la structure apprenante sont alors figées. Les `CELL` existantes peuvent encore reconnaître et produire le `readout`, mais aucune mémoire apprenante ne change, aucun seed, aucune promotion et aucune nouvelle `LAYER` ne sont créés. En modes `temporal` et `predictive`, le registre causal du pas précédent continue néanmoins d'avancer conformément au §5.9 : le gel de l'apprentissage n'arrête pas l'ordre des présentations.

Une population apprenante n'avance son horloge que lorsqu'elle reçoit une présentation non vide dans son propre espace. En modes `temporal` et `predictive`, l'espace géométrique d'une `LAYER` et son espace temporel associé possèdent donc des horloges d'apprentissage indépendantes. L'absence de présentation temporelle ne provoque aucun oubli temporel.

### 1.4 Mode causal

Le paramètre de construction :

\[
\boxed{mode\in\{\texttt{geometry},\texttt{temporal},\texttt{predictive}\}}
\]

vaut `geometry` par défaut.

`mode` appartient à la configuration causale persistante et est immuable pour un état existant. Toute autre valeur est invalide.

- `geometry` : seule la cognition dans `E` existe ;
- `temporal` : `geometry` inchangé, plus la cognition de succession dans `T(E)=E\oplus E` ;
- `predictive` : `temporal` inchangé, plus la projection éphémère `présent reconnu → successeur connu possible` définie au §5.12.

Il n'existe aucun drapeau prédictif indépendant : la prédiction implique nécessairement la cognition temporelle.

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

La reconnaissance est éphémère. Elle ne modifie pas la géométrie avant la phase d'apprentissage et participe à la fois au `readout` externe et au contexte vertical du §5.

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

## 5. Contexte reconnu, readout et récursion

### 5.1 Valeurs reconnues d'un atome

Pour chaque atome présenté `X_s`, définir l'ensemble exact des valeurs reconnues :

\[
\boxed{
R_s=\{C_i^-:\ i\in I_s\}/=,
}
\]

où `/=` quotient les centres vectoriellement exactement identiques.

Poser :

\[
\boxed{n_s=|R_s|.}
\]

Les identités administratives, les supports `A_i`, les dispersions `V_i` des `CELL` et les responsabilités `\rho_{is}` n'appartiennent pas à la géométrie du contexte une fois `R_s` déterminé.

Si `n_s=0`, l'atome ne contribue pas au contexte reconnu.

Si `n_s>0`, chaque valeur `c\in R_s` contribue au contexte par le noyau ponctuel :

\[
\boxed{
\left(\frac{r_s}{n_s},c,0\right).
}
\]

Cette répartition est uniforme entre les valeurs reconnues distinctes d'un même atome. Elle conserve la masse de l'atome sans introduire d'autorité d'identité, de support ou d'ordre.

### 5.2 Noyau de contexte d'une LAYER

Pour toute la présentation reçue par une `LAYER`, fusionner par la loi du §2.1 toutes les contributions du §5.1 :

\[
\boxed{
H_L^{\uparrow}
=
\bigoplus_{s:n_s>0}
\bigoplus_{c\in R_s}
\left(\frac{r_s}{n_s},c,0\right).
}
\]

S'il n'existe aucune reconnaissance, `H_L^{\uparrow}` est absent.

Sinon écrire :

\[
\boxed{
H_L^{\uparrow}
=
(W_L^{\uparrow},C_L^{\uparrow},V_L^{\uparrow}).
}
\]

Sa masse vérifie exactement :

\[
\boxed{
W_L^{\uparrow}
=
\sum_{s:n_s>0}r_s
\le|\mathcal P|.
}
\]

`C_L^{\uparrow}` est le barycentre des connaissances effectivement reconnues pendant la présentation, avec conservation de la masse causale des atomes.

`V_L^{\uparrow}` mesure la dispersion **entre les valeurs reconnues**. Cette dispersion appartient au contexte reconnu lui-même ; elle n'est ni une erreur d'explication, ni une mémoire d'autorité des `CELL`.

Le noyau de contexte dépend donc seulement :

- des masses de la présentation ;
- des ensembles exacts de valeurs reconnues par ses atomes.

Conditionnellement à ces ensembles, il est indépendant des supports EMA des `CELL`, de leurs responsabilités d'apprentissage, de leurs identités et de leur ordre.

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

- `V_L^{\uparrow}=0` signifie que toute la reconnaissance de la présentation se réduit à une seule valeur vectorielle distincte ; aucune relation entre connaissances distinctes n'est donc formée ;
- `C_L^{\uparrow}=0` signifie que le contexte ne possède aucune direction vectorielle canonique ; il reste silencieux.

Aucune direction arbitraire n'est construite pour sauver un contexte exactement centré en zéro.

Lorsqu'il est émissible, le contexte de la `LAYER` suivante est exactement la présentation singleton :

\[
\boxed{
\operatorname{input}(L_{k+1})
=
\{H_{L_k}^{\uparrow}\}.
}
\]

Il n'existe donc jamais d'arbre de branches inter-couches : une `LAYER` émet au plus un noyau de contexte par présentation.

### 5.4 Limite de résolution contextuelle

Le contrat vertical conserve exactement le quotient `(W,C,V)` du contexte reconnu. Deux configurations de reconnaissances distinctes produisant exactement le même noyau contextuel sont indiscernables pour les couches supérieures.

Cette perte est native au type cognitif d'Auxein : aucune covariance, orientation de second ordre ou identité de constituant n'est transmise.

En particulier, une relation parfaitement symétrique de centre nul, telle qu'un contexte constitué de `+a` et `-a` à masses égales, n'a aucun représentant vectoriel non nul compatible avec l'invariance orthogonale. Elle reste silencieuse.

### 5.5 Readout géométrique du NETWORK

Chaque instance reçoit une étiquette d'univers :

\[
\boxed{u_N\in\mathrm{String}^+.}
\]

`u_N` est une chaîne non vide, égale à `"auxein"` par défaut. Elle identifie le contexte sémantique extérieur de l'instance et n'intervient dans aucune décision cognitive interne.

Pour tout triplet `(k,s,i)` tel que la `CELL i` de `L_k` reçoit une responsabilité positive sur le noyau présenté :

\[
X_{ks}=(r_{ks},c_{ks},v_{ks}),
\]

produire la reconnaissance éphémère :

\[
\boxed{R_{ksi}=(u_N,c_{ks},C_{ki}^-).}
\]

Sa représentation externe canonique reste le triplet ordonné JSON-compatible :

```text
[universe, local_input, recognised]
```

La dispersion interne `v_{ks}` ne fait pas partie de l'identité externe d'une reconnaissance. Elle participe à l'admissibilité interne, pas à la valeur vectorielle reconnue.

Le contexte géométrique externe du pas est l'ensemble exact des reconnaissances produites sur toutes les `LAYER` effectivement parcourues :

\[
\boxed{
\mathcal C_t
=
\{(u_N,c_{ks},C_{ki}^-):\rho_{kis}>0\}.
}
\]

Deux occurrences de triplets exactement identiques constituent la même reconnaissance et sont coalescées sans multiplicité.

`\mathcal C_t` ne contient ni indice de `LAYER`, ni identité de `CELL`, ni masse, ni responsabilité, ni provenance. Il est dérivé, éphémère, n'est jamais relu par Auxein et n'appartient pas à l'état persistant. En mode `geometry`, `\mathcal C_t` est exactement le `readout` retourné par le `NETWORK`.

### 5.6 Récursion du NETWORK

Le `NETWORK` est une suite ordonnée :

```text
L0 → L1 → L2 → ...
```

`L0` reçoit la présentation extérieure uniformisée du §1.1.

Pour chaque `LAYER` suivante qui existait déjà au début du pas, elle reçoit l'unique noyau de contexte émissible produit par la couche précédente. Si aucun contexte émissible n'est produit, aucune couche supérieure n'est parcourue pour cette branche causale ; il n'existe qu'une branche.

Une `LAYER` sans `CELL` ne produit aucun contexte vertical. Elle apprend uniquement les noyaux reçus dans `Σ_L`.

### 5.7 Croissance verticale

Si une `LAYER` terminale produit un contexte émissible et qu'aucune `LAYER` suivante n'existe, la géométrie demande la création d'une nouvelle `LAYER` vide, seulement si `β>0`.

Cette création appartient à la transaction globale du §7.3. Si elle est refusée, l'état cognitif existant reste inchangé. Le contexte courant n'est rejoué ni mémorisé hors de toute `LAYER`.

Une `LAYER` créée pendant le pas ne lit pas le contexte qui a provoqué sa création. Une nouvelle profondeur exige donc au moins une nouvelle occurrence future du contexte.

### 5.8 Espace temporel associé à une LAYER

En modes `temporal` et `predictive`, pour chaque `LAYER L_k` de monde `E=\mathbb R^D`, le `NETWORK` possède deux populations finies dans :

\[
\boxed{T(E)=E\oplus E\simeq\mathbb R^{2D}.}
\]

Elles sont :

\[
\boxed{\Sigma_k^T}
\]

et une population de `CELL` temporelles.

Ces populations utilisent sans modification les lois des §2, §3 et §4, avec dimension `2D`. Elles sont associées à `L_k` mais n'appartiennent pas à sa cognition géométrique : `L_k` ne les lit jamais.

Les deux espaces sont strictement étanches :

- une `CELL` géométrique ne concerne jamais une présentation temporelle ;
- une `CELL` temporelle ne concerne jamais une présentation géométrique ;
- les deux populations ne partagent ni `Σ`, ni allocation, ni promotion, ni contexte cognitif ;
- aucune `CELL` temporelle ne participe au contexte vertical du §5.2 ni à la croissance verticale du §5.7.

Leur seule coexistence est structurelle dans le même `NETWORK`, économique au §7 et externe dans le `readout` des §5.12–5.13.

### 5.9 Registre précédent et présentation temporelle

Pour chaque `L_k`, le `NETWORK` entretient en modes `temporal` et `predictive` un registre causal :

\[
\boxed{P_k\in\{\varnothing\}\cup\{(W,C,V):W>0,\ C\in E,\ V\ge0\}.}
\]

`P_k` contient exactement le noyau de contexte `H_{L_k}^{\uparrow}` produit au pas extérieur précédent, lorsqu'il existait. Il n'existe aucun historique au-delà de ce registre unique.

Soient, pour deux pas extérieurs strictement consécutifs :

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

et

\[
\boxed{
\|(x_-,x_+)-(C_-,C_+)\|^2
=
\|x_--C_-\|^2+\|x_+-C_+\|^2.
}
\]

Aucune covariance entre les deux extrémités n'est requise.

La présentation temporelle reçue par l'espace associé à `L_k` est le singleton :

\[
\boxed{\mathcal P_{k,t}^T=\{X_{k,t}^T\}.}
\]

Le noyau `H_{L_k,t}^{\uparrow}` est utilisable ici dès qu'il existe, indépendamment de l'autorité verticale du §5.3. Un singleton reconnu (`V=0`) ou un contexte de centre nul reste donc un état temporel valide.

Si `P_k` ou `H_{L_k,t}^{\uparrow}` est absent, aucune présentation temporelle n'existe pour `L_k` à ce pas. L'absence ne vaut jamais présentation nulle.

Après le traitement temporel du pas :

\[
\boxed{
P_k\leftarrow
\begin{cases}
H_{L_k,t}^{\uparrow},&\text{si ce noyau existe},\\
\varnothing,&\text{sinon}.
\end{cases}
}
\]

Cette mise à jour s'applique à toutes les `LAYER` existantes au début du pas. Une `LAYER` non parcourue ou ne produisant aucune reconnaissance obtient donc `P_k=\varnothing`; aucune transition `t-2\to t` ne peut être fabriquée.

Le registre avance également à `eta=0`. Une `LAYER` créée pendant le pas commence avec `P_k=\varnothing`.

### 5.10 Cognition temporelle

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

avec les mêmes lois et les mêmes frontières causales que dans `E`.

Une `CELL` temporelle possède donc exactement un noyau :

\[
\boxed{H_j^T=(A_j^T,C_j^T,V_j^T),\qquad C_j^T\in E\oplus E.}
\]

Écrire, uniquement comme projections géométriques :

\[
C_j^T=C_{j,-}^T\oplus C_{j,+}^T.
\]

`C_{j,-}^T` et `C_{j,+}^T` ne sont ni des identités, ni des pointeurs, ni des références vers des `CELL` géométriques.

Une connaissance temporelle représente exclusivement une succession adjacente. Les `CELL` temporelles ne produisent aucun noyau destiné à un autre espace temporel :

\[
\boxed{T(T(E))\text{ n'appartient pas à Auxein}.}
\]

### 5.11 Limite de résolution temporelle

Le contrat temporel conserve exactement le quotient :

\[
\boxed{
(W_-W_+,\ C_-\oplus C_+,\ V_-+V_+).
}
\]

Il ne conserve pas séparément `V_-` et `V_+`. Deux transitions produisant exactement le même noyau temporel sont cognitivement indistinguables.

`CONCERN` s'applique au couple complet dans `E\oplus E`. Une meilleure correspondance sur une extrémité peut donc compenser une moins bonne correspondance sur l'autre dans la distance quadratique totale. Il n'existe aucun test canonique séparé `CONCERN(source) ∧ CONCERN(target)`.

Si :

\[
C_-=0\qquad\land\qquad C_+=0,
\]

alors le centre temporel vaut exactement zéro. Conformément aux §2.4 et §4.2, cette présentation peut faire avancer l'horloge temporelle mais ne peut être reconnue, alimenter `Σᵀ` ni créer une connaissance. Aucune direction artificielle n'est construite pour distinguer `0→0`.

### 5.12 Projection prédictive

Cette section s'applique uniquement en mode `predictive`. Elle ne définit aucune nouvelle mémoire ni aucune nouvelle population.

Soit le noyau de contexte géométrique reconnu courant d'une `LAYER` :

\[
H_{L_k,t}^{\uparrow}=(W_t,C_t,V_t),
\]

lorsqu'il existe, y compris s'il est verticalement silencieux. Pour toute `CELL` temporelle **préexistante au début de sa frontière temporelle du pas** :

\[
H_j^T=(A_j^T,C_{j,-}^T\oplus C_{j,+}^T,V_j^T),
\]

le `NETWORK` considère uniquement les deux noyaux ponctuels dérivés :

\[
\boxed{X_t^P=(1,C_t,0),\qquad H_{j,-}^P=(1,C_{j,-}^T,0).}
\]

La `CELL` temporelle `j` est prédictivement concernée si et seulement si le `CONCERN` canonique du §2.4 appliqué à ces deux noyaux ponctuels est vrai, soit exactement :

\[
\boxed{
\|C_t-C_{j,-}^T\|^2<\|C_t\|^2
\quad\land\quad
\|C_t-C_{j,-}^T\|^2<\|C_{j,-}^T\|^2.
}
\]

Ni `A_j^T` ni `V_j^T` ne participent à ce test. Le quotient temporel du §5.11 ne conserve pas séparément les dispersions source et cible ; reconstruire un rayon source depuis `V_j^T` inventerait donc une information absente. La projection prédictive porte canoniquement sur le centre seulement.

Pour toute `CELL` ainsi concernée, produire l'élément prédictif éphémère :

\[
\boxed{
Q_{ktj}=(u_N,C_t,C_{j,-}^T,C_{j,+}^T).
}
\]

Sa représentation externe JSON-compatible est :

```text
[universe, current_context, recognised_source, predicted_successor]
```

Poser :

\[
\boxed{\mathcal Q_t=\{Q_{ktj}:\text{concernement prédictif}\}.}
\]

Deux quadruplets exactement identiques sont coalescés sans multiplicité. Aucun indice de `LAYER`, identité de `CELL`, support, variance, responsabilité ou provenance n'est exposé.

Conséquences normatives :

- si plusieurs `CELL` concernées partagent la même source mais des successeurs distincts, tous les successeurs sont émis ;
- une projection source `C_{j,-}^T=0` ne peut concerner aucun présent ;
- une projection cible `C_{j,+}^T=0` est une prédiction explicite valide et reste distincte d'une absence de prédiction ;
- `Σ_k^T` ne prédit jamais : seule une connaissance devenue `CELL` possède cette autorité ;
- une `CELL` temporelle créée ou promue pendant le pas ne peut prédire qu'à partir du pas suivant ;
- aucune prédiction n'est relue par Auxein, ne modifie `P_k`, ne forme une présentation temporelle et ne déclenche une prédiction de profondeur supérieure.

Ainsi, si `A→B` et `B→C` sont connus, un présent `A` peut émettre `B` mais jamais `C` par fermeture transitive au même pas. La prédiction canonique porte exclusivement sur `step → step+1`.

### 5.13 Readout temporel, prédictif et contexte global du pas

Pour toute `CELL` temporelle `j` concernée par l'atome :

\[
X_{k,t}^T=(r^T,c_-\oplus c_+,v^T),
\]

produire la reconnaissance éphémère :

\[
\boxed{
S_{ktj}
=
(u_N,(c_-,c_+),(C_{j,-}^{T,-},C_{j,+}^{T,-})).
}
\]

Sa représentation externe JSON-compatible est :

```text
[universe, [previous_input, current_input], [previous_recognised, current_recognised]]
```

Comme pour le readout géométrique, masses, dispersions, responsabilités, indices de `LAYER` et identités de `CELL` n'appartiennent pas à cette reconnaissance. Deux triplets temporels exactement identiques sont coalescés sans multiplicité.

Poser :

\[
\boxed{
\mathcal S_t
=
\{S_{ktj}:\text{responsabilité temporelle positive}\}.
}
\]

En mode `geometry` :

\[
\boxed{\operatorname{readout}_{N,t}=\mathcal C_t.}
\]

En mode `temporal`, le contexte global externe du pas est le couple typé :

\[
\boxed{
\operatorname{readout}_{N,t}
=
(\mathcal C_t,\mathcal S_t).
}
\]

Sa représentation JSON-compatible canonique est :

```text
{
  "concepts":   [...],
  "sequences":  [...]
}
```

En mode `predictive` :

\[
\boxed{
\operatorname{readout}_{N,t}
=
(\mathcal C_t,\mathcal S_t,\mathcal Q_t).
}
\]

Sa représentation JSON-compatible canonique est :

```text
{
  "concepts":     [...],
  "sequences":    [...],
  "predictions":  [...]
}
```

Ces listes ne sont jamais fusionnées vectoriellement. Leur appartenance au même `readout` exprime uniquement leur coexistence à la même frontière causale extérieure. Le `readout` ne crée aucun lien persistant ni communication cognitive entre ces vues.

---

## 6. Causalité d'une présentation

À toute population effectivement présentée sont associés trois états conceptuels :

\[
\boxed{
X^-\xrightarrow{\text{perception unique}}X^*
\xrightarrow{\text{normalisation}}X^+.
}
\]

Pour une `LAYER`, ces états sont ceux du compartiment géométrique. En modes `temporal` et `predictive`, la population temporelle associée possède sa propre frontière de présentation et applique exactement la même discipline de snapshot. La lecture prédictive du mode `predictive` observe le snapshot temporel préexistant et ne constitue pas une population apprenante.

Tous les `CONCERN`, `ALLOCATE`, reconnaissances, contextes, cibles EMA et décisions privées d'un espace sont calculés exclusivement depuis son snapshot `X^-` et sa présentation courante. Aucun objet créé pendant le pas ne peut lire, concerner, apprendre, être reconnu ou émettre pour ce même pas. Aucun replay n'existe.

Pour chaque présentation extérieure :

1. restaurer d'abord la solvabilité matérielle si nécessaire (§7.4) ; toute contraction forcée invalide simultanément tous les registres `P_k` ;
2. figer la suite des `LAYER` existantes pour ce pas et initialiser les contextes éphémères `\mathcal C_t`, en modes `temporal` et `predictive` `\mathcal S_t`, et en mode `predictive` `\mathcal Q_t` ;
3. construire la présentation uniforme du §1.1 et la remettre à `L0` ;
4. pour chaque `LAYER` existante recevant une présentation géométrique non vide, dans l'ordre du réseau :
   1. coalescer les atomes de géométrie exactement identique `(c,v)` ;
   2. figer le snapshot géométrique ;
   3. appliquer `CONCERN/ALLOCATE` aux `CELL` géométriques du snapshot ;
   4. produire les reconnaissances de `\mathcal C_t` et le noyau `H_L^{\uparrow}` depuis ces mêmes `CELL` ;
   5. si `H_L^{\uparrow}` est verticalement émissible et que la `LAYER` suivante existait au début du pas, lui transmettre immédiatement la présentation singleton `{H_L^{\uparrow}}` ;
   6. mettre à jour exactement une fois les `CELL` géométriques préexistantes ;
   7. appliquer `DETECT` aux seuls atomes géométriques inconnus depuis le `Σ_L` du snapshot, puis mettre à jour exactement une fois ses composantes préexistantes ;
   8. normaliser le compartiment géométrique selon le §4.4 ;
   9. si la `LAYER` est terminale, que son contexte était émissible, qu'aucun successeur n'existait et que `β>0`, former une demande de nouvelle `LAYER` ;
5. en modes `temporal` et `predictive`, après la phase géométrique complète, pour chaque `LAYER` qui existait au début du pas :
   1. prendre le registre `P_k` du snapshot de pas et le noyau géométrique `H_{L_k,t}^{\uparrow}` produit à ce pas, éventuellement absents ;
   2. figer le snapshot des `CELL` temporelles préexistantes ; en mode `predictive`, si le contexte courant existe, produire `\mathcal Q_t` depuis ce snapshot selon le §5.12 ;
   3. si `P_k` et le contexte courant existent, construire `\mathcal P_{k,t}^T` selon le §5.9 ;
   4. si cette présentation existe, appliquer `CONCERN/ALLOCATE` depuis le même snapshot temporel, produire les reconnaissances de `\mathcal S_t`, mettre à jour exactement une fois les `CELL` temporelles et `Σ_k^T`, puis normaliser ce compartiment selon les §3 et §4 ;
   5. remplacer `P_k` par le noyau courant ou par `∅` selon le §5.9 ;
6. réunir tous les seeds géométriques survivants, tous les seeds temporels survivants et l'éventuelle demande de `LAYER` en une transaction globale de croissance (§7.3) ;
7. exécuter cette transaction entière si elle est payable, sinon ne rien créer ;
8. créer, pour toute nouvelle `LAYER` admise en modes `temporal` ou `predictive`, son compartiment temporel vide et son registre `P_k=∅` sans lui faire lire le pas courant ni prédire ;
9. terminer la matérialisation du `readout` selon le mode, puis retourner le `readout` et l'état post-pas.

La phase temporelle ne peut modifier aucune reconnaissance géométrique du même pas. La phase géométrique ne lit aucun état temporel. Leur seul raccord causal apprenant est la construction par le `NETWORK` de `\mathcal P_{k,t}^T` depuis deux contextes géométriques successifs. En mode `predictive`, la projection du §5.12 est une lecture supplémentaire unidirectionnelle du snapshot temporel vers le `readout` seulement.

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

- `format_version=4`, `dimension`, `steps_seen`, `layer_count` sur `u64` ;
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

En modes `temporal` et `predictive`, chaque `LAYER` possède :

- quatre compteurs `u64` pour `Σ_L`, `CELL`, `Σ_k^T`, `CELL^T` ;
- un tag de présence du registre `P_k` ;
- un slot fixe de noyau géométrique `U_H` réservé à `P_k`, qu'il soit présent ou absent.

Ainsi :

\[
\boxed{U_L^T=33+U_H.}
\]

La réservation fixe de `P_k` garantit que l'observation d'un nouveau contexte précédent ne constitue jamais une croissance matérielle hors transaction. Le contenu absent du slot n'a aucune autorité cognitive.

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

En modes `temporal` et `predictive` :

\[
\boxed{
M_{units}(\mathcal A)
=
U_N+
\sum_L
\left[
U_L^T
+(|\Sigma_L|+N_C(L))U_H
+(|\Sigma_L^T|+N_T(L))U_T
\right].
}
\]

Le contexte vertical, les présentations temporelles, les projections prédictives et le `readout` sont éphémères et ne possèdent aucun coût persistant propre. Le mode `predictive` a donc exactement la même empreinte matérielle persistante que `temporal` pour un état de connaissance identique.

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
U_L^T,&mode\in\{\texttt{temporal},\texttt{predictive}\}.
\end{cases}
}
\]

L'état minimal exécutable est `NETWORK + L0` vide :

\[
\boxed{
M_{min}=
\begin{cases}
U_N+U_L^G,&mode=\texttt{geometry},\\
U_N+U_L^T,&mode\in\{\texttt{temporal},\texttt{predictive}\}.
\end{cases}
}
\]

Si `B_{units}<M_{min}`, l'environnement est inexécutable.

Une interface peut exprimer ergonomiquement le budget en unités de noyau, mais toute décision interne utilise exclusivement `B_{units}`.

### 7.3 Croissance

Les promotions du §4.4 sont géométriques et matériellement neutres ; elles sont appliquées avant toute création matérielle.

Après lecture géométrique et, en modes `temporal` et `predictive`, temporelle du pas, réunir :

- toutes les demandes de nouveaux noyaux `Σ_L` encore admissibles du §4.4 ;
- en modes `temporal` et `predictive`, toutes les demandes de nouveaux noyaux `Σ_k^T` encore admissibles ;
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

1. vider simultanément toutes les mémoires `Σ_L` et, en modes `temporal` et `predictive`, toutes les mémoires `Σ_k^T` ;
2. supprimer toute `LAYER` terminale sans aucune `CELL` géométrique ni temporelle, sans jamais supprimer `L0` ;
3. si l'état reste insolvable, considérer ensemble les valeurs distinctes `K_i` de toutes les `CELL` géométriques et temporelles restantes et, pour chaque valeur `k`, l'état `\mathcal A_{>k}` obtenu en conservant exactement les `CELL` des deux espaces telles que `K_i>k`, puis en supprimant les `LAYER` terminales devenues sans connaissance ;
4. s'il existe un `k` tel que `M_{units}(\mathcal A_{>k})\le B_{units}`, choisir le plus petit et remplacer l'état par `\mathcal A_{>k}` ;
5. sinon, si l'état minimal `NETWORK + L0` vide est solvable, supprimer toutes les `CELL` des deux espaces et ramener le réseau à cet état minimal ; sinon l'environnement est inexécutable ;
6. si au moins une contraction a été nécessaire, poser simultanément `P_k=\varnothing` pour toutes les `LAYER` survivantes.

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

Après perception, les EMA, promotions, suppressions de travail couvert, coalescences et mises à jour des slots `P_k` n'augmentent pas l'empreinte ; la seule croissance persistante est `G_t`, soumise à un unique test de payabilité.

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
7. conservation de la masse des responsabilités ;
8. conservation de la masse contextuelle `W_L^{\uparrow}=Σ_{s:n_s>0}r_s` ;
9. indépendance cognitive des `LAYER` hors présentation `(r,c,v)` ;
10. absence de replay ;
11. absence de subdivision causale d'une présentation par une optimisation d'exécution ;
12. absence d'autorité des supports `A_i` et des responsabilités `ρ_i` dans la géométrie du contexte vertical ;
13. unicité du noyau contextuel émis par couche et par présentation ;
14. silence vertical d'une reconnaissance réduite à une seule valeur distincte ;
15. silence vertical d'un contexte exactement centré en zéro ;
16. distinction de l'ordre temporel : en général `A→B` et `B→A` occupent des centres distincts dans `E⊕E` ;
17. absence de transition à travers un pas sans contexte reconnu dans la `LAYER` considérée ;
18. indépendance cognitive stricte des populations géométriques et temporelles ;
19. absence de récursion `T(T(E))` ;
20. même changement d'échelle uniforme dans les deux extrémités temporelles, avec `V^T→a²V^T` ;
21. même rotation orthogonale appliquée aux deux extrémités temporelles ;
22. même économie de contraction pour des `CELL` de valeur `K` égale, quel que soit leur espace ;
23. la projection prédictive commute avec tout changement d'échelle uniforme non nul ;
24. la projection prédictive commute avec toute rotation orthogonale appliquée simultanément au présent, à la source et au successeur ;
25. une dispersion temporelle différente à centre temporel identique ne modifie pas la lecture prédictive.

L'origine `0` est sémantique ; une translation uniforme n'est donc pas une invariance exigée.

Une égalité géométrique exacte ne peut être résolue par un ID, une adresse mémoire, un ordre de conteneur ou un axe arbitraire.

Une masse nulle n'apprend rien. Une `CELL` avec `C_i=0` ne concerne aucun atome. Un contexte avec `C^{\uparrow}=0` n'est pas émis.

Deux contextes distincts ayant exactement le même quotient `(W,C,V)` sont cognitivement indistinguables pour la couche suivante. Auxein n'invente aucune structure supplémentaire pour les séparer.

Deux successions distinctes ayant exactement le même quotient `(W_-W_+,C_-⊕C_+,V_-+V_+)` sont de même cognitivement indistinguables pour l'espace temporel. La dispersion temporelle n'est pas décomposable en dispersions d'extrémité.

Une succession dont les deux centres valent exactement zéro possède un centre temporel nul et reste silencieuse. En mode `predictive`, une source temporelle projetée exactement nulle reste également silencieuse, tandis qu'une cible projetée nulle peut être émise explicitement.

### 8.1 Calcul numérique

Les valeurs persistantes utilisent `f32` ou `f64`. Les calculs intermédiaires doivent être réalisés au moins en `binary64` avant projection atomique dans le format persistant.

Toute décision qui porte sur la validité de l'état persistant s'applique à la valeur effectivement projetée. En particulier, une demande de seed est projetée, renormalisée et retestée contre les `CELL` au §7.3 avant d'entrer dans `Σ`.

Les réductions dont l'ordre n'a aucune autorité doivent être reproductibles et indépendantes de l'ordre d'itération.

Les variances et fusions utilisent les formes positives du §2 ; aucune soustraction de grands moments presque égaux n'est nécessaire à la loi canonique.

Aucune valeur seulement petite ne peut être remplacée par zéro au moyen d'un epsilon comportemental. Les zéros structurellement démontrés peuvent être construits exactement.

Le test `V_L^{\uparrow}=0` signifie que toutes les valeurs reconnues distinctes fusionnées ont exactement la même position après quotient ; aucune tolérance ne crée ni ne détruit une relation verticale.

Une implémentation doit empêcher qu'un support positif soit interprété comme une destruction cognitive uniquement à cause d'un sous-flux numérique lors de l'oubli. Une renormalisation commune ou toute représentation mathématiquement équivalente est admissible si elle conserve exactement les décisions canoniques.

### 8.2 Frontière d'implémentation

Caches, index, décroissances différées, queues, parallélisme, chunking, mémoïsation et structures de travail sont autorisés s'ils sont entièrement reconstructibles depuis l'état canonique et n'altèrent aucune décision.

Un index géométrique peut réduire les candidats aux concernements publics et privés, mais seuls les prédicats du §2.4 possèdent l'autorité.

La construction du contexte peut être incrémentale, mais elle doit être exactement équivalente à la fusion commutative des contributions du §5.1.

Une décroissance différée doit distinguer l'horloge de chaque espace effectivement présenté. En modes `temporal` et `predictive`, une présentation géométrique sans présentation temporelle ne fait pas vieillir les mémoires temporelles associées. Une lecture prédictive seule ne fait jamais avancer leur horloge.

Une présentation reste un événement causal unique quelle que soit sa réalisation physique.

---

## 9. État persistant canonique

L'état persistant est minimal.

### 9.1 NETWORK

- ordre des `LAYER` ;
- `format_version=4` administratif ;
- `dimension` ;
- `scalar∈{f32,f64}` ;
- `memory` ;
- `eta` ;
- `mode∈{geometry,temporal,predictive}` ;
- compteur de présentations achevées ;
- en modes `temporal` et `predictive`, un registre `P_k` présent ou absent pour chaque `LAYER`.

Le budget appartient à l'environnement matériel et n'est pas une connaissance apprise.

L'étiquette d'univers du `readout` appartient à l'interface et n'est pas une mémoire cognitive.

Les registres `P_k` sont un état causal du `NETWORK`, pas une connaissance. Ils sont néanmoins persistants : sauvegarder puis recharger entre deux pas doit préserver exactement la succession `step-1→step`.

### 9.2 LAYER

Toujours :

- `Σ_L`, population finie de noyaux `(W,C,V)` en dimension `D` ;
- population de `CELL` géométriques.

En modes `temporal` et `predictive`, le `NETWORK` entretient en outre, structurellement associé à cette `LAYER` :

- `Σ_k^T`, population finie de noyaux `(W,C,V)` en dimension `2D` ;
- population de `CELL` temporelles.

Ces populations temporelles ne constituent pas un quatrième niveau architectural et ne sont jamais lues par la `LAYER`.

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
- projections et sorties prédictives courantes ;
- responsabilités ;
- ensembles `R_s` ;
- noyaux de contexte `H_L^{\uparrow}` du pas courant après transfert éventuel dans `P_k` ;
- `readout` ;
- demandes de croissance non encore commises ;
- caches et index d'exécution.

---

## 10. Fermeture

Pour un état canonique fini `\mathcal A_t`, une présentation extérieure finie `\mathcal X_t`, une configuration causale valide et un environnement matériel exécutable, les sections précédentes définissent une transition finie et un contexte reconnu éphémère :

\[
\boxed{
(\mathcal A_t,\mathcal X_t;u_N)
\longmapsto
(\mathcal A_{t+1},\operatorname{readout}_{N,t}).
}
\]

Le noyau cognitif géométrique d'une `LAYER` est :

```text
présentation de noyaux
→ CELL concernées
→ reconnaissances
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

En modes `temporal` et `predictive`, le `NETWORK` ajoute strictement :

```text
contexte reconnu à step-1
+ contexte reconnu à step
→ noyau produit dans E⊕E
→ CELL temporelles concernées
→ inconnu vers Σᵀ
→ récurrence
→ CELL temporelle
```

En mode `predictive`, le `NETWORK` ajoute ensuite, sans apprentissage :

```text
contexte reconnu à step
+ projection source d'une CELL temporelle préexistante
→ CONCERN ponctuel
→ projection successeur
→ readout prédictif de step+1 possible
```

La croissance horizontale, la croissance verticale, l'apprentissage temporel et la lecture prédictive sont donc distincts :

\[
\boxed{
\text{inconnu récurrent}
\longrightarrow
\text{nouvelle connaissance dans le même espace},
}
\]

\[
\boxed{
\text{connaissances distinctes reconnues dans une même présentation}
\longrightarrow
\text{contexte de la LAYER suivante},
}
\]

\[
\boxed{
H_{k,t-1}^{\uparrow},H_{k,t}^{\uparrow}
\longrightarrow
\text{présentation temporelle dans }T(E_k).
}
\]

\[
\boxed{
C_{k,t}^{\uparrow},\ C_{j,-}^T\oplus C_{j,+}^T
\longrightarrow
C_{j,+}^T\text{ si }C_{j,-}^T\text{ concerne ponctuellement }C_{k,t}^{\uparrow}.
}
\]

Les quatre primitives cognitives nommées restent :

```text
CONCERN
ALLOCATE
DETECT
CONTEXT
```

`CONCERN` et `ALLOCATE` utilisent l'unique primitive de population du §2.4. `DETECT` applique cette même géométrie privément à `Σ`, dans `E` comme dans `T(E)`. `CONTEXT` fusionne uniquement les valeurs géométriques reconnues pour la récursion verticale. La construction du noyau temporel et la projection prédictive sont des responsabilités structurelles du `NETWORK`, pas de nouvelles lois cognitives.

Une abstraction supérieure est une régularité récurrente du contexte compact des connaissances déjà reconnues par l'étage précédent. Une connaissance temporelle est une régularité récurrente entre deux de ces contextes strictement adjacents. Elle ne possède ni histoire plus profonde, ni pointeur vers ses extrémités conceptuelles, ni autorité verticale. En mode `predictive`, sa projection gauche peut seulement servir de clé géométrique ponctuelle pour exposer sa projection droite comme futur connu possible.

En mode `temporal`, le `readout` réunit sans les mélanger le contexte géométrique du pas et le contexte de succession reconnu entre le pas précédent et le pas courant. En mode `predictive`, il y ajoute les successeurs possibles lus depuis le présent reconnu. Ces réunions sont externes et éphémères.

La géométrie détermine les connaissances présentes, les concernements et les créations admissibles. Le `NETWORK` observe l'ordre causal des pas. L'économie maintient un état fini sans sélectionner arbitrairement entre créations simultanées. Une connaissance acquise persiste indépendamment de son actualité et ne peut être perdue que lorsqu'une contraction matérielle est obligatoire, selon sa valeur géométrique intrinsèque.
