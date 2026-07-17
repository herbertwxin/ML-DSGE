<TeXmacs|2.1.5>

<style|generic>

<\body>
  <section|Standard RBC>

  Agent solves\ 

  <\equation>
    max <rsub|><big|sum><rsub|t=1><rsup|\<infty\>>\<beta\><rsup|t>\<bbb-E\><rsub|t>
    <frac|c<rsub|t><rsup|1-\<gamma\>>|1-\<gamma\>>,\<forall\>t=1\<ldots\>\<infty\>
  </equation>

  subject to\ 

  <\equation*>
    c<rsub|t>+k<rsub|t+1>=A<rsub|t> k<rsub|t><rsup|\<alpha\>>+<around*|(|1-\<delta\>|)>k<rsub|t>
  </equation*>

  The first-order conditions are

  <\align*>
    <tformat|<table|<row|<cell|\<beta\><rsup|t>c<rsub|t><rsup|-\<gamma\>>=>|<cell|\<lambda\><rsub|t>>>|<row|<cell|\<lambda\><rsub|t>=>|<cell|\<bbb-E\><rsub|t
    >\<lambda\><rsub|t+1><around*|(|A<rsub|t+1>
    k<rsub|t+1><rsup|\<alpha\>-1>+1-\<delta\>|)>>>>>
  </align*>

  So our Euler equation is\ 

  <\equation>
    <frac|1|c<rsub|t><rsup|\<gamma\>>>=\<bbb-E\><rsub|t><frac|1|c<rsub|t+1><rsup|\<gamma\>>><around*|(|A<rsub|t+1>
    k<rsub|t+1><rsup|\<alpha\>-1>+1-\<delta\>|)>
  </equation>

  <section|RBC with labor >

  Agent solves\ 

  <\equation>
    max <rsub|><big|sum><rsub|t=1><rsup|\<infty\>>\<beta\><rsup|t>\<bbb-E\><rsub|t><frac|c<rsub|t><rsup|1-\<gamma\>>|1-\<gamma\>>-\<chi\>\<cdot\><frac|n<rsub|t><rsup|1+<frac|1|\<nu\>>>|1+<frac|1|\<nu\>>>
  </equation>

  subject to\ 

  <\equation*>
    c<rsub|t>+k<rsub|t+1>=A<rsub|t> k<rsub|t><rsup|\<alpha\>>n<rsub|t<rsup|>><rsup|1-\<alpha\>>+<around*|(|1-\<delta\>|)>k<rsub|t>
  </equation*>

  So the first-order conditions are

  <\align*>
    <tformat|<table|<row|<cell|\<beta\><rsup|t>c<rsub|t><rsup|-\<gamma\>>=>|<cell|\<lambda\><rsub|t>>>|<row|<cell|\<chi\>
    n<rsub|t><rsup|<frac|1|\<nu\>>>=>|<cell|\<lambda\><rsub|t>
    <around*|(|1-\<alpha\>|)>A<rsub|t> k<rsub|t><rsup|\<alpha\>>n<rsub|t><rsup|-\<alpha\>>>>|<row|<cell|\<lambda\><rsub|t>=>|<cell|\<bbb-E\><rsub|t>\<lambda\><rsub|t+1><around*|(|A<rsub|t+1><rsub|>
    k<rsub|t+1><rsup|\<alpha\>-1>n<rsub|t><rsup|\<alpha\>>+1-\<delta\>|)>>>>>
  </align*>

  So the 2 optimal conditions are

  <\align*>
    <tformat|<table|<row|<cell| n<rsub|t><rsup|<frac|1|\<nu\>>>=>|<cell|<frac|1|\<chi\>>\<cdot\>
    c<rsub|t><rsup|-\<gamma\>> <around*|(|1-\<alpha\>|)>A<rsub|t>
    k<rsub|t><rsup|\<alpha\>>n<rsub|t><rsup|-\<alpha\>>>>|<row|<cell|c<rsub|t><rsup|-\<gamma\>>=>|<cell|\<bbb-E\><rsub|t>
    c<rsub|t+1><rsup|-\<gamma\>><around*|(|A<rsub|t+1><rsub|>
    k<rsub|t+1><rsup|\<alpha\>-1>n<rsub|t+1><rsup|\<alpha\>-1>+1-\<delta\>|)>>>>>
  </align*>

  Since the intratemporal condition is uniquely pin down by the choice of
  <math|c<rsub|t>>,

  <\equation*>
    n<rsub|t><rsup|<frac|1|\<gamma\>>+\<alpha\>>=<frac|1|\<chi\>>\<cdot\>
    c<rsub|t><rsup|-\<gamma\>> <around*|(|1-\<alpha\>|)>A<rsub|t>
    k<rsub|t><rsup|\<alpha\>>
  </equation*>

  We can write the Euler equation solely in <math|c<rsub|t>>

  <\equation>
    <tabular|<tformat|<table|<row|<cell|c<rsub|t><rsup|-\<gamma\>>=>|<cell|\<bbb-E\><rsub|t>
    c<rsub|t+1><rsup|-\<gamma\>><around*|[|A<rsub|t+1><rsub|>
    k<rsub|t+1><rsup|\<alpha\>-1><around*|(|<frac|1|\<chi\>>\<cdot\>
    c<rsub|t+1><rsup|-\<gamma\>> <around*|(|1-\<alpha\>|)>A<rsub|t+1>
    k<rsub|t+1><rsup|\<alpha\>>|)><rsub|><rsup|\<alpha\>-1<around*|(|<frac|1|\<gamma\>>+\<alpha\>|)>>+1-\<delta\>|]>>>>>>
  </equation>
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1|../../../../.TeXmacs/texts/scratch/no_name_1.tm>>
    <associate|auto-2|<tuple|2|?|../../../../.TeXmacs/texts/scratch/no_name_1.tm>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Standard
      RBC> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>