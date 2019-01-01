
\ProvidesClass{roadef2018}[2009/06/19 Classe personnelle, V1.0, modifie le 2011/09/21 par J.E. Mendoza]
 \typeout{-------------------------------------------------------}
 \typeout{Classe ROADEF 2018}
 \typeout{-------------------------------------------------------}
% classe de base

\LoadClass[a4paper,11pt,twoside]{article}


% extensions

\RequirePackage[latin1]{inputenc}
\RequirePackage[T1]{fontenc}
% \RequirePackage[francais]{babel}
\RequirePackage[english]{babel}
\RequirePackage{lmodern}
\RequirePackage{graphicx}
\RequirePackage{epsfig}
\RequirePackage{amssymb}


\renewcommand{\fnum@figure}{\small{FIG.~\thefigure}}
\renewcommand{\fnum@table}{\small{TAB.~\thetable}}



\renewcommand{\normalsize}{\fontsize{11pt}{12pt}\selectfont}


\setlength{\textheight}{250mm}				% hauteur texte
\setlength{\topmargin}{-20mm}				% espace au dessus de en-tete
\setlength{\headheight}{12pt}				% hauteur en-tete
\setlength{\headsep}{15mm}					% espace entre en_tete et texte
\setlength{\footskip}{0mm}					%  espace entre texte et pied de page


% largeur
% page impaire : 25+5+155+25=210mm
% page paire : 25+155+5+25=210mm
\setlength{\textwidth}{155mm}        	% largeur texte
\setlength{\oddsidemargin}{5mm}		% marge gauche pages impaires
\setlength{\evensidemargin}{0mm}	% marge gauche pages paires




\setlength{\parindent}{1em}				% indentation
%\setlength{\parskip}{1em}					% interparagraphe
\renewcommand{\baselinestretch}{1.1}					% interligne
\flushbottom
%\raggedbottom


% \renewcommand{\abstract}[1]{\textbf{Résumé} : \itshape  }
\newcommand{\keywords}[1]{\textbf{Keywords} : \itshape #1 \normalfont}



\newtheorem{remarque}{Remarque}
\newtheorem{propriete}{Propriété}
\newtheorem{corollaire}{Corollaire}
\newtheorem{definition}{Définition}
\newtheorem{notation}{Notation}
\newtheorem{theoreme}{Théorème}
\newtheorem{proposition}{Proposition}
\newtheorem{exemple}{Exemple}

\newenvironment{preuve}{ \nopagebreak \noindent\textbf{Preuve~: }\nopagebreak }{ \vspace{-0.1cm}\hfill $\Box$ \vspace{-0.2cm}}

\AtEndOfClass{
\typeout{-------------------------------------------------------}
 \typeout{Classe ROADEF 2018}
 \typeout{-------------------------------------------------------}
 }

\newcommand{\email}[1]{{\tt#1}}

\newcounter{@inst}

\newcommand{\inst}[1]{\unskip$^{#1}$}

\newcommand{\institute}[1]{\gdef\@institute{#1}}

\newcommand{\institutename}{\par
 \begingroup
 \parskip=\z@
 \parindent=\z@
 \setcounter{@inst}{1}%
 \def\and{\par\stepcounter{@inst}%
 \noindent$^{\the@inst}$\enspace\ignorespaces}%
 \setbox0=\vbox{\def\thanks##1{}\@institute}%
 \ifnum\c@@inst=1\relax
   \gdef\fnnstart{0}%
 \else
   \xdef\fnnstart{\c@@inst}%
   \setcounter{@inst}{1}%
   \noindent$^{\the@inst}$\enspace
 \fi
 \ignorespaces
 \@institute\par
 \endgroup}




\renewcommand\@maketitle{%
  \newpage
  \thispagestyle{empty}
  \null
  \begin{center}%
    {\LARGE	\@title \par}%
    \vskip 3em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
  \par
  \vskip 0.5em
 {\small\institutename}
  \end{center}%
%    \markboth{ ROADEF 2018 - Troyes}{ \shorttitle}

  }





  % définition de l'en-tête

  \def\ps@myheadings{%
    \let\@oddfoot\@empty\let\@evenfoot\@empty
    \def\@evenhead{\hfil\slshape\leftmark}%
    \def\@oddhead{{\slshape\rightmark\hfil}}%
    \let\@mkboth\@gobbletwo
    \let\sectionmark\@gobble
    \let\subsectionmark\@gobble
    }




 \pagestyle{myheadings}





