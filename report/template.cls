% ~~~ identification ~~~~~~~~~
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{template}[2025/08/25 MA580 Homework Problems]
% ~~~ initial code ~~~~~~~
\LoadClass[12pt]{article}
\usepackage{graphicx}
\usepackage[left=1in, right=1in, top=1in, bottom=1in]{geometry}
\usepackage{titlesec}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{enumerate}
\usepackage{enumitem}
\usepackage[pdfborderstyle={/S/U/W 1}]{hyperref}
\usepackage{parskip}
\usepackage{cancel}
\usepackage{algpseudocode}
\usepackage{algorithm}
% \usepackage{amsthm}
\usepackage{subcaption}


% \usepackage[english]{babel}
% \usepackage[autostyle, english = american]{csquotes}
% \MakeOuterQuote{"}


\usepackage{minted}

\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
% \renewcommand\qedsymbol{QED}
\renewcommand\theoremautorefname{Lemma}
\renewcommand{\sectionautorefname}{Sec.}

\setlength{\parindent}{0em}
\setlength{\parskip}{1ex}
% ~~~ declaration of options ~~~~~~~~~
\renewcommand\thesubsection{\thesection.\arabic{subsection}}
\titleformat{\section}{\normalfont\Large\bfseries}{\thesection. }{1ex}{}
\titleformat{\subsection}[runin]{\normalfont\large\bfseries}{\newline\newline\hspace{2em}\thesubsection}{1ex}{}[\setlength{\parindent}{2em}]




\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\trace}{\mathrm{trace}}
\newcommand{\mat}[1]{\mathbf{{#1}}}
\newcommand*{\defeq}{\stackrel{\text{def}}{=}}
\newcommand{\norm}[1]{\left\| {#1} \right\|}
\newcommand{\ip}[2]{\left\langle{#1}, {#2}\right\rangle}
% \renewcommand{\vec}[1]{{\mathchoice
%                      {\mbox{\boldmath$\displaystyle{#1}$}}
%                      {\mbox{\boldmath$\textstyle{#1}$}}
%                      {\mbox{\boldmath$\scriptstyle{#1}$}}
%                      {\mbox{\boldmath$\scriptscriptstyle{#1}$}}}}