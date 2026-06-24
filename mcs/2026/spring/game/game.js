// ============================================================
//  BRAINCADE — Game Logic
// ============================================================

const AVATARS = ["🦊","🐼","🐸","🦁","🐯","🐵","🐶","🐱","🐰","🐲","🦄","🐙","🦖","🐢","🦉","🐬","🦋","🐝","🦜","🐧","🦒","🐨","🐷","🐮"];
const PCOLORS = ["#e21b3c","#1368ce","#26890c","#d89e00","#9b5de5","#f15bb5","#00bbf9","#fb5607","#06d6a0","#ef476f","#118ab2","#8338ec"];

const state = {
  players: [],          // {name, avatar, color, score, lastDelta, streak}
  deck: null,           // {title,color,emoji,questions:[]}
  deckId: null,
  questions: [],        // flat list: one distinct question per player-slot
  cursor: 0,            // flat index into questions (advances every answer)
  rounds: 8,            // number of rounds (each player answers 1 Q per round)
  roundNum: 0,          // current round (0-based)
  turn: 0,              // which player answers within the current round
  qCount: 8,            // selected rounds-per-player (0 = as many as deck allows)
  timeLimit: 20,        // seconds (0 = off)
  timer: null,
  timeLeft: 0,
  answered: false,
};

// ---------- Navigation ----------
function go(id){
  document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
  document.getElementById('screen-'+id).classList.add('active');
  window.scrollTo({top:0,behavior:'smooth'});
  if(id==='decks') renderDecks();
  // themed background only shows during gameplay
  if(typeof hideThemeBg==='function' && id!=='play') hideThemeBg();
  // music: gameplay uses the topic theme (set in renderQuestion); everywhere
  // else uses the menu theme. SFX: a soft navigation whoosh.
  if(typeof AudioFX!=='undefined'){
    if(id!=='play') AudioFX.playMusic('menu');
    AudioFX.play('nav');
  }
}

// Fallback: map a lesson title back to its id (used if a question lacks lessonId)
function lessonIdFromTitle(title){
  const l = (typeof LESSONS!=='undefined') && LESSONS.find(x=>x.title===title);
  return l ? l.id : null;
}

// ---------- Players ----------
function addPlayer(){
  const inp = document.getElementById('nameInput');
  let name = inp.value.trim();
  if(!name) return;
  if(state.players.length >= 12){ flashInput("Max 12 players"); return; }
  if(state.players.some(p=>p.name.toLowerCase()===name.toLowerCase())){ flashInput("Name taken"); return; }
  const idx = state.players.length;
  state.players.push({
    name,
    avatar: AVATARS[idx % AVATARS.length],
    color: PCOLORS[idx % PCOLORS.length],
    score:0, lastDelta:0, streak:0
  });
  inp.value=""; inp.focus();
  if(typeof AudioFX!=='undefined') AudioFX.play('add');
  renderPlayers();
}
function flashInput(msg){
  const inp=document.getElementById('nameInput');
  const old=inp.placeholder; inp.value=""; inp.placeholder=msg;
  inp.style.borderColor="#dc2626";
  setTimeout(()=>{inp.placeholder=old;inp.style.borderColor="";},1200);
}
function removePlayer(i){
  if(typeof AudioFX!=='undefined') AudioFX.play('remove');
  state.players.splice(i,1);
  // reassign avatars/colors to keep them tidy
  state.players.forEach((p,idx)=>{p.avatar=AVATARS[idx%AVATARS.length];p.color=PCOLORS[idx%PCOLORS.length];});
  renderPlayers();
}
function renderPlayers(){
  const wrap=document.getElementById('players');
  const hint=document.getElementById('emptyHint');
  wrap.innerHTML="";
  state.players.forEach((p,i)=>{
    const d=document.createElement('div');
    d.className='player-card'; d.style.background=p.color;
    d.innerHTML=`<span class="av">${p.avatar}</span><span class="nm">${escapeHtml(p.name)}</span>
                 <button class="rm" title="Remove" onclick="removePlayer(${i})">✕</button>`;
    wrap.appendChild(d);
  });
  hint.style.display = state.players.length? "none":"block";
  document.getElementById('toDeckBtn').disabled = state.players.length < 2;
}
document.addEventListener('keydown',e=>{
  if(e.key==='Enter' && document.getElementById('screen-players').classList.contains('active')) addPlayer();
});

// ---------- Decks ----------
function renderDecks(){
  const grid=document.getElementById('deckGrid');
  grid.innerHTML="";
  LESSONS.forEach(l=>{
    const d=document.createElement('div');
    d.className='deck'; d.style.background=`linear-gradient(135deg,${l.color},${shade(l.color,-25)})`;
    d.dataset.id=l.id;
    d.innerHTML=`<span class="day">Day ${l.day}</span><div class="em">${l.emoji}</div>
                 <h3>${l.title}</h3><p>${l.subtitle}</p>
                 <div class="cnt">${l.questions.length} questions</div>`;
    d.onclick=()=>selectDeck(l.id,d);
    grid.appendChild(d);
  });
  // Mixed deck
  const m=document.createElement('div');
  m.className='deck mixed'; m.dataset.id='mixed';
  const total=LESSONS.reduce((s,l)=>s+l.questions.length,0);
  m.innerHTML=`<span class="day">Mix</span><div class="em">🌈</div>
               <h3>All Lessons</h3><p>A random mix from every topic — the ultimate challenge.</p>
               <div class="cnt">${total} questions in the pool</div>`;
  m.onclick=()=>selectDeck('mixed',m);
  grid.appendChild(m);

  setupSegments();
  // re-apply selection highlight if any
  if(state.deckId){
    const el=grid.querySelector(`.deck[data-id="${state.deckId}"]`);
    if(el) el.classList.add('sel');
    document.getElementById('startBtn').disabled=false;
  } else {
    document.getElementById('startBtn').disabled=true;
  }
}
function selectDeck(id,el){
  state.deckId=id;
  document.querySelectorAll('.deck').forEach(d=>d.classList.remove('sel'));
  el.classList.add('sel');
  document.getElementById('startBtn').disabled=false;
  if(typeof AudioFX!=='undefined') AudioFX.play('select');
}
let segReady=false;
function setupSegments(){
  if(segReady) return; segReady=true;
  document.querySelectorAll('#segCount button').forEach(b=>{
    b.onclick=()=>{document.querySelectorAll('#segCount button').forEach(x=>x.classList.remove('on'));b.classList.add('on');state.qCount=+b.dataset.n; if(typeof AudioFX!=='undefined') AudioFX.play('click');};
  });
  document.querySelectorAll('#segTime button').forEach(b=>{
    b.onclick=()=>{document.querySelectorAll('#segTime button').forEach(x=>x.classList.remove('on'));b.classList.add('on');state.timeLimit=+b.dataset.t; if(typeof AudioFX!=='undefined') AudioFX.play('click');};
  });
}

// ---------- Start ----------
function startGame(){
  // build question pool
  let pool;
  if(state.deckId==='mixed'){
    pool = buildMixedDeck();
    state.deck={title:"All Lessons",color:"#7209b7",emoji:"🌈"};
  } else {
    const l=LESSONS.find(x=>x.id===state.deckId);
    pool = l.questions.map(q=>({...q,lessonId:l.id,lessonTitle:l.title,lessonColor:l.color,emoji:l.emoji}));
    state.deck={title:l.title,color:l.color,emoji:l.emoji};
  }
  const nPlayers = state.players.length;
  const basePool = shuffle(pool.slice());

  // Each player gets their OWN distinct question every round.
  // rounds = chosen question count (per player); 0 = use as many rounds as
  // the deck allows (one full pass of unique questions across all players).
  let rounds;
  if(state.qCount>0) rounds = state.qCount;
  else rounds = Math.max(1, Math.floor(basePool.length / nPlayers));
  const needed = rounds * nPlayers;

  // Build a flat list of `needed` questions. Cycle through a freshly shuffled
  // copy of the deck whenever we run out, so every slot is filled and
  // repeats are spread apart.
  const flat = [];
  let bag = shuffle(basePool.slice());
  while(flat.length < needed){
    if(bag.length===0) bag = shuffle(basePool.slice());
    flat.push(bag.shift());
  }

  // shuffle answer order per question, tracking the correct index
  state.questions = flat.map(q=>{
    const order = shuffle([0,1,2,3]);
    const newAnswers = order.map(i=>q.a[i]);
    const newCorrect = order.indexOf(q.correct);
    return {...q, a:newAnswers, correct:newCorrect};
  });

  state.rounds = rounds;

  // reset scores
  state.players.forEach(p=>{p.score=0;p.lastDelta=0;p.streak=0;});
  state.roundNum = 0;     // 0-based current round
  state.turn = 0;         // which player answers within the round
  state.cursor = 0;       // flat index into state.questions
  state.answered = false;
  go('play');
  renderQuestion();
}

// The question for the current player this round is at the flat cursor.
function currentQ(){ return state.questions[state.cursor]; }

// ---------- Gameplay ----------
const SHAPES=["▲","◆","●","■"];
function renderQuestion(){
  clearTimer();
  state.answered=false;
  const q=currentQ();
  const player=state.players[state.turn];

  document.getElementById('deckPill').textContent=`${state.deck.emoji} ${state.deck.title}`;
  document.getElementById('progPill').textContent=`Round ${state.roundNum+1} / ${state.rounds}`;

  const who=document.getElementById('turnWho');
  who.textContent=`${player.avatar} ${player.name}`;
  who.style.background=player.color;

  document.getElementById('qnLabel').textContent=`Player ${state.turn+1} of ${state.players.length} · their question`;
  const tag=document.getElementById('qTag');
  tag.textContent=`${q.emoji} ${q.lessonTitle}`;
  tag.style.background=hexToRgba(q.lessonColor,.22);

  // topic-relevant animated background + matching theme music
  const themeId = q.lessonId || lessonIdFromTitle(q.lessonTitle);
  setThemeBg(themeId, q.lessonColor);
  if(typeof AudioFX!=='undefined') AudioFX.playMusic(themeId);

  document.getElementById('qText').textContent=q.q;

  const ansWrap=document.getElementById('answers');
  ansWrap.innerHTML="";
  q.a.forEach((txt,i)=>{
    const b=document.createElement('button');
    b.className=`ans a${i}`;
    b.innerHTML=`<span class="shape">${SHAPES[i]}</span><span>${escapeHtml(txt)}</span>`;
    b.onclick=()=>pickAnswer(i,b);
    ansWrap.appendChild(b);
  });

  document.getElementById('feedback').classList.remove('show');
  renderMiniLb();
  startTimer();
}

function startTimer(){
  const arc=document.getElementById('timerArc');
  const num=document.getElementById('timerNum');
  const ring=document.getElementById('timerRing');
  const C=175.9;
  if(state.timeLimit<=0){
    ring.style.display="none";
    state.timeLeft=0; state.startTime=Date.now();
    return;
  }
  ring.style.display="block";
  state.timeLeft=state.timeLimit; state.startTime=Date.now();
  num.textContent=state.timeLeft;
  arc.style.strokeDashoffset=0;
  arc.style.stroke="#ffd60a";
  state.timer=setInterval(()=>{
    state.timeLeft--;
    num.textContent=Math.max(0,state.timeLeft);
    const frac=state.timeLeft/state.timeLimit;
    arc.style.strokeDashoffset=C*(1-frac);
    if(frac<=0.33) arc.style.stroke="#ff405b";
    else if(frac<=0.6) arc.style.stroke="#ffb300";
    if(state.timeLeft<=0){ clearTimer(); timeUp(); }
  },1000);
}
function clearTimer(){ if(state.timer){clearInterval(state.timer);state.timer=null;} }

function pickAnswer(i,btn){
  if(state.answered) return;
  state.answered=true;
  clearTimer();
  if(typeof AudioFX!=='undefined') AudioFX.play('click');
  const q=currentQ();
  const player=state.players[state.turn];
  const correct = (i===q.correct);

  // scoring: speed bonus
  let pts=0;
  if(correct){
    let speedFrac=1;
    if(state.timeLimit>0){
      const elapsed=(Date.now()-state.startTime)/1000;
      speedFrac=Math.max(0, 1-(elapsed/state.timeLimit));
    } else {
      speedFrac=0.6; // flat when timer off
    }
    pts = Math.round(500 + 500*speedFrac); // 500–1000
    player.streak++;
    if(player.streak>=2) pts += (player.streak-1)*100; // streak bonus
  } else {
    player.streak=0;
  }
  player.score += pts;
  player.lastDelta = pts;

  if(typeof AudioFX!=='undefined') AudioFX.play(correct?'correct':'wrong');

  // reveal answers
  document.querySelectorAll('.ans').forEach((b,idx)=>{
    b.disabled=true;
    if(idx===q.correct) b.classList.add('correct');
    else if(idx===i) b.classList.add('wrong-pick');
    else b.classList.add('wrong');
  });

  showFeedback(correct,pts,q,player,false,i);
  renderMiniLb();
}
function timeUp(){
  if(state.answered) return;
  state.answered=true;
  const q=currentQ();
  const player=state.players[state.turn];
  player.streak=0; player.lastDelta=0;
  document.querySelectorAll('.ans').forEach((b,idx)=>{
    b.disabled=true;
    if(idx===q.correct) b.classList.add('correct'); else b.classList.add('wrong');
  });
  document.getElementById('timerNum').textContent="0";
  if(typeof AudioFX!=='undefined') AudioFX.play('timeup');
  showFeedback(false,0,q,player,true,-1);
}

function showFeedback(correct,pts,q,player,timedOut,pickedIdx){
  const fb=document.getElementById('feedback');
  const v=document.getElementById('fbVerdict');
  const p=document.getElementById('fbPts');
  const w=document.getElementById('fbWhy');
  if(correct){
    v.textContent = pickPraise(player.streak);
    v.className="verdict good";
    p.textContent=`+${pts} points`+(player.streak>=2?`  🔥 ${player.streak} in a row!`:``);
    burstConfetti(0.6);
    w.innerHTML=`<b>Correct answer:</b> ${escapeHtml(q.a[q.correct])}<br>${escapeHtml(q.why)}`;
  } else {
    v.textContent = timedOut? "⏰ Time's up!" : "Not quite!";
    v.className="verdict bad";
    p.textContent="+0 points";
    // Wrong-answer explainer: name what they picked and why it's not right,
    // then give the correct answer and the teaching note.
    let html='';
    if(!timedOut && pickedIdx>=0){
      html += `<div class="wrong-line">❌ You picked: <b>${escapeHtml(q.a[pickedIdx])}</b><br>`
            + `<span class="wrong-note">${escapeHtml(wrongExplainer(q,pickedIdx))}</span></div>`;
    } else if(timedOut){
      html += `<div class="wrong-line">⏰ No answer was chosen in time.</div>`;
    }
    html += `<div class="right-line">✅ Correct answer: <b>${escapeHtml(q.a[q.correct])}</b></div>`
          + `<div class="why-note">${escapeHtml(q.why)}</div>`;
    w.innerHTML = html;
  }
  fb.classList.add('show');
  // next button label
  document.getElementById('nextBtn').textContent =
    isLastPlayerOfRound()? "See Leaderboard 📊" : "Next Player ▶";
  fb.scrollIntoView({behavior:'smooth',block:'nearest'});
}

function pickPraise(streak){
  if(streak>=4) return "🔥 UNSTOPPABLE!";
  if(streak>=3) return "⭐ On Fire!";
  if(streak>=2) return "💪 Nice streak!";
  const opts=["✅ Correct!","🎉 Yes!","👏 Great!","🙌 Right!"];
  return opts[Math.floor(Math.random()*opts.length)];
}

// Brief, topic-aware explainer for a WRONG pick. Uses a per-question
// wrongHints[idx] when the author provided one, otherwise builds a helpful
// line from the lesson topic so every wrong answer gets a short coaching note.
const TOPIC_NUDGE = {
  "Robot Maze & Search": "In the maze, think about the start (S), goal (G), walls (#), open dots (.), and how the robot remembers where it has been.",
  "Tiny Neural Networks": "Remember the neuron's parts: weight, bias, target, error = target − prediction, and the update w = w + lr × error × x.",
  "Recommendation Systems": "Recall the picker: each vote is a 1 or 0, the average is sum ÷ count, and the 'top' item has the highest average.",
  "AI for Images": "Picture the pixel grid: an edge is a sudden number change, and the filter compares a pixel to its 4 neighbors.",
  "Chatbots & Language AI": "Think about the chatbot loop: receive → lowercase/strip → match a keyword/intent → reply, and how that can misfire.",
  "Data, Bias & Fairness": "Consider how the training data was collected — imbalance or one-sided data leads to biased, unfair predictions."
};
function wrongExplainer(q, idx){
  if(q.wrongHints && q.wrongHints[idx]) return q.wrongHints[idx];
  return `That option doesn't fit here. ${TOPIC_NUDGE[q.lessonTitle] || "Re-read the question and match it to today's key idea."}`;
}

// Is the current player the last one in this round?
function isLastPlayerOfRound(){
  return state.turn === state.players.length-1;
}
function isLastRound(){
  return state.roundNum >= state.rounds-1;
}

// ---------- Turn / round flow ----------
// Flow: every ROUND, each player answers their OWN distinct question (one per
// player, in turn). After the last player of the round answers -> leaderboard.
function nextStep(){
  if(!isLastPlayerOfRound()){
    state.turn++;
    state.cursor++;     // next player gets the next (different) question
    renderQuestion();
  } else {
    // everyone has answered their question this round -> leaderboard
    showScoreboard();
  }
}

function showScoreboard(){
  const last = isLastRound();
  document.getElementById('scoresTitle').textContent = last? "Final Standings 🏁":"Leaderboard 📊";
  document.getElementById('scoresSub').textContent = `After round ${state.roundNum+1} of ${state.rounds}`;
  const sorted=[...state.players].sort((a,b)=>b.score-a.score);
  const list=document.getElementById('lbList');
  list.innerHTML="";
  sorted.forEach((p,i)=>{
    const row=document.createElement('div');
    row.className='lb-row'+(i===0?' lead':'');
    const medal = i===0?"🥇":i===1?"🥈":i===2?"🥉":(i+1);
    row.innerHTML=`<div class="rk">${medal}</div><div class="av">${p.avatar}</div>
      <div class="nm">${escapeHtml(p.name)}</div>
      <div class="sc">${p.score}${p.lastDelta>0?`<span class="delta">+${p.lastDelta}</span>`:``}</div>`;
    list.appendChild(row);
  });
  document.getElementById('continueBtn').textContent = last? "🏆 See Final Results":"Next Round ▶";
  go('scores');
}

function afterScores(){
  if(isLastRound()){
    showResults();
  } else {
    state.roundNum++;
    state.turn=0;
    state.cursor++;     // advance into the next round's first question
    go('play');
    renderQuestion();
  }
}

function renderMiniLb(){
  const wrap=document.getElementById('miniLb');
  wrap.innerHTML="";
  [...state.players].sort((a,b)=>b.score-a.score).forEach(p=>{
    const c=document.createElement('div');
    c.className='chip';
    c.innerHTML=`<span>${p.avatar}</span><span>${escapeHtml(p.name)}</span><span class="sc">${p.score}</span>`;
    wrap.appendChild(c);
  });
}

// ---------- Results / Podium ----------
function showResults(){
  const sorted=[...state.players].sort((a,b)=>b.score-a.score);
  const winner=sorted[0];
  // handle tie for first
  const topScore=sorted[0].score;
  const winners=sorted.filter(p=>p.score===topScore);
  document.getElementById('winnerLine').innerHTML =
    winners.length>1
      ? `🤝 It's a tie at <b>${topScore}</b> points!`
      : `${winner.avatar} <b>${escapeHtml(winner.name)}</b> wins with <b>${winner.score}</b> points!`;

  // podium: positions 2,1,3 layout
  const pod=document.getElementById('podium');
  pod.innerHTML="";
  const layout=[
    {p:sorted[1],cls:'second',rank:2},
    {p:sorted[0],cls:'first',rank:1},
    {p:sorted[2],cls:'third',rank:3},
  ];
  layout.forEach(item=>{
    if(!item.p) return;
    const el=document.createElement('div');
    el.className='pod '+item.cls;
    el.innerHTML=`<div class="ava">${item.p.avatar}</div>
      <div class="pname">${escapeHtml(item.p.name)}</div>
      <div class="pscore">${item.p.score} pts</div>
      <div class="bar">${item.rank===1?'🥇':item.rank===2?'🥈':'🥉'}</div>`;
    pod.appendChild(el);
  });

  const rest=document.getElementById('restList');
  rest.innerHTML="";
  sorted.slice(3).forEach((p,i)=>{
    const row=document.createElement('div');
    row.className='lb-row';
    row.innerHTML=`<div class="rk">${i+4}</div><div class="av">${p.avatar}</div>
      <div class="nm">${escapeHtml(p.name)}</div><div class="sc">${p.score}</div>`;
    rest.appendChild(row);
  });

  go('results');
  bigConfetti();
  if(typeof AudioFX!=='undefined') AudioFX.play('fanfare');
}

// ---------- Replays ----------
function playAgainSameDeck(){ startGame(); }

// ---------- Quit to Main Menu (in-app modal) ----------
function quitToHome(){
  clearTimer();
  document.getElementById('modalOverlay').classList.add('show');
}
function closeQuitModal(){
  document.getElementById('modalOverlay').classList.remove('show');
  // if we were mid-question with no answer yet, resume the timer
  const onPlay = document.getElementById('screen-play').classList.contains('active');
  if(onPlay && !state.answered) startTimer();
}
function confirmQuit(){
  document.getElementById('modalOverlay').classList.remove('show');
  clearTimer();
  // reset scores/progress so the next game starts clean (players are kept)
  state.players.forEach(p=>{p.score=0;p.lastDelta=0;p.streak=0;});
  state.roundNum=0; state.turn=0; state.cursor=0; state.answered=false;
  confetti=[];
  go('home');
}
function fullReset(){ clearTimer(); go('home'); }

// Allow Esc to open the Main Menu prompt during play/scoreboard
document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){
    const onPlay=document.getElementById('screen-play').classList.contains('active');
    const onScores=document.getElementById('screen-scores').classList.contains('active');
    const modalOpen=document.getElementById('modalOverlay').classList.contains('show');
    if(modalOpen){ closeQuitModal(); }
    else if(onPlay||onScores){ quitToHome(); }
  }
});

// ---------- Utilities ----------
function shuffle(arr){ for(let i=arr.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[arr[i],arr[j]]=[arr[j],arr[i]];} return arr; }
function escapeHtml(s){ return String(s).replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }
function hexToRgba(hex,a){ const n=parseInt(hex.slice(1),16); return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`; }
function shade(hex,pct){ const n=parseInt(hex.slice(1),16);let r=(n>>16)&255,g=(n>>8)&255,b=n&255;
  r=Math.max(0,Math.min(255,r+Math.round(255*pct/100)));g=Math.max(0,Math.min(255,g+Math.round(255*pct/100)));b=Math.max(0,Math.min(255,b+Math.round(255*pct/100)));
  return '#'+((1<<24)+(r<<16)+(g<<8)+b).toString(16).slice(1); }

// ---------- Confetti ----------
const cvs=document.getElementById('confetti'); const cx=cvs.getContext('2d');
let confetti=[];
function resizeC(){cvs.width=innerWidth;cvs.height=innerHeight;} resizeC(); addEventListener('resize',resizeC);
const CCOL=["#ffd60a","#ff5ca8","#4cc9f0","#43c91f","#9b5de5","#fb5607"];
function burstConfetti(amount=1){
  const n=Math.round(60*amount);
  for(let i=0;i<n;i++){
    confetti.push({x:innerWidth/2+(Math.random()-0.5)*200,y:innerHeight/2,vx:(Math.random()-0.5)*9,vy:-Math.random()*11-4,
      g:0.28,size:Math.random()*8+4,col:CCOL[i%CCOL.length],rot:Math.random()*6,vr:(Math.random()-0.5)*0.4,life:90});
  }
  if(!confAnim) loopConf();
}
function bigConfetti(){
  for(let i=0;i<220;i++){
    confetti.push({x:Math.random()*innerWidth,y:-20-Math.random()*200,vx:(Math.random()-0.5)*4,vy:Math.random()*4+2,
      g:0.12,size:Math.random()*9+5,col:CCOL[i%CCOL.length],rot:Math.random()*6,vr:(Math.random()-0.5)*0.4,life:260});
  }
  if(!confAnim) loopConf();
}
let confAnim=false;
function loopConf(){
  confAnim=true;
  cx.clearRect(0,0,cvs.width,cvs.height);
  confetti.forEach(p=>{p.vy+=p.g;p.x+=p.vx;p.y+=p.vy;p.rot+=p.vr;p.life--;
    cx.save();cx.translate(p.x,p.y);cx.rotate(p.rot);cx.fillStyle=p.col;
    cx.fillRect(-p.size/2,-p.size/2,p.size,p.size*0.6);cx.restore();});
  confetti=confetti.filter(p=>p.life>0 && p.y<cvs.height+40);
  if(confetti.length){requestAnimationFrame(loopConf);}else{cx.clearRect(0,0,cvs.width,cvs.height);confAnim=false;}
}

// ---------- Topic-themed animated background ----------
// A single canvas that draws a motif relevant to the current question's
// lesson. Re-themes whenever the topic changes; fades out off the play screen.
const tb = document.getElementById('themebg');
const tctx = tb.getContext('2d');
function resizeTB(){ tb.width=innerWidth; tb.height=innerHeight; }
resizeTB(); addEventListener('resize', resizeTB);

let tbTheme=null, tbAgents=[], tbT=0, tbColor='#ffffff', tbRAF=null;

function setThemeBg(lessonId, color){
  tbColor = color || '#ffffff';
  if(tbTheme===lessonId){ tb.classList.add('show'); return; }
  tbTheme = lessonId;
  buildThemeAgents(lessonId);
  tb.classList.add('show');
  if(!tbRAF) tbRAF = requestAnimationFrame(drawThemeBg);
}
function hideThemeBg(){ tb.classList.remove('show'); }

function rnd(a,b){ return a+Math.random()*(b-a); }
function buildThemeAgents(id){
  tbAgents=[];
  const W=tb.width, H=tb.height;
  if(id==='maze'){
    const cell=Math.max(54, Math.min(86, W/16));
    for(let y=cell/2; y<H; y+=cell) for(let x=cell/2; x<W; x+=cell){
      const r=Math.random();
      tbAgents.push({x,y,cell,kind:r<0.06?'S':r<0.12?'G':r<0.28?'#':'.',ph:Math.random()*6});
    }
    tbAgents.push({mover:true,x:rnd(0,W),y:rnd(0,H),tx:rnd(0,W),ty:rnd(0,H),cell});
  } else if(id==='neuron'){
    const layers=[3,4,4,2]; const lx=W*0.18, gap=(W*0.64)/(layers.length-1);
    layers.forEach((n,li)=>{ for(let i=0;i<n;i++){ tbAgents.push({node:true,x:lx+li*gap,y:H*(i+1)/(n+1),li,ph:Math.random()*6}); } });
  } else if(id==='recommend'){
    for(let i=0;i<22;i++) tbAgents.push({x:rnd(0,W),y:rnd(0,H),vy:rnd(0.2,0.7),sym:Math.random()<0.5?'\uD83D\uDC4D':'\u2B50',size:rnd(20,40),sp:rnd(0,6)});
  } else if(id==='images'){
    const cell=Math.max(34,Math.min(56,W/26));
    for(let y=0;y<H;y+=cell) for(let x=0;x<W;x+=cell) tbAgents.push({x,y,cell,ph:Math.random()*6,sp:rnd(0.5,1.5)});
  } else if(id==='chatbot'){
    for(let i=0;i<16;i++) tbAgents.push({x:rnd(0,W),y:rnd(0,H),vy:rnd(0.15,0.5),size:rnd(34,64),sp:rnd(0,6)});
  } else if(id==='bias'){
    const n=10, bw=W/n;
    for(let i=0;i<n;i++) tbAgents.push({x:i*bw+bw*0.18,bw:bw*0.64,ph:Math.random()*6,sp:rnd(0.4,1.0)});
  }
}

function drawThemeBg(){
  tbRAF=requestAnimationFrame(drawThemeBg);
  tbT+=0.016;
  const W=tb.width, H=tb.height;
  tctx.clearRect(0,0,W,H);
  if(!tb.classList.contains('show')) return;
  const col=tbColor;
  tctx.lineWidth=2;

  if(tbTheme==='maze'){
    tbAgents.forEach(a=>{
      if(a.mover){
        a.x+=(a.tx-a.x)*0.02; a.y+=(a.ty-a.y)*0.02;
        if(Math.hypot(a.tx-a.x,a.ty-a.y)<8){ a.tx=rnd(0,W); a.ty=rnd(0,H); }
        tctx.fillStyle=hexToRgba(col,.9);
        tctx.beginPath(); tctx.arc(a.x,a.y,7,0,Math.PI*2); tctx.fill();
        return;
      }
      const s=a.cell*0.62, x=a.x-s/2, y=a.y-s/2;
      if(a.kind==='#'){ tctx.fillStyle=hexToRgba(col,.16); tctx.fillRect(x,y,s,s); }
      else if(a.kind==='S'){ tctx.fillStyle=hexToRgba(col,.5); tctx.fillRect(x,y,s,s); tbLabel('S',a.x,a.y,s); }
      else if(a.kind==='G'){ tctx.fillStyle=hexToRgba(col,.5); tctx.fillRect(x,y,s,s); tbLabel('G',a.x,a.y,s); }
      else { tctx.strokeStyle=hexToRgba(col,.12); tctx.strokeRect(x,y,s,s); }
    });
  } else if(tbTheme==='neuron'){
    tbAgents.filter(n=>n.node).forEach(a=>{
      tbAgents.filter(b=>b.node && b.li===a.li+1).forEach(b=>{
        const pulse=0.08+0.08*(0.5+0.5*Math.sin(tbT*2+a.x*0.01+b.y*0.01));
        tctx.strokeStyle=hexToRgba(col,pulse);
        tctx.beginPath(); tctx.moveTo(a.x,a.y); tctx.lineTo(b.x,b.y); tctx.stroke();
      });
    });
    tbAgents.filter(n=>n.node).forEach(a=>{
      const r=10+2*Math.sin(tbT*3+a.ph);
      tctx.fillStyle=hexToRgba(col,.55);
      tctx.beginPath(); tctx.arc(a.x,a.y,r,0,Math.PI*2); tctx.fill();
    });
  } else if(tbTheme==='recommend' || tbTheme==='chatbot'){
    tbAgents.forEach(a=>{
      a.y-=a.vy; if(a.y<-60){ a.y=H+40; a.x=rnd(0,W); }
      const wob=Math.sin(tbT+a.sp)*10;
      if(tbTheme==='recommend'){
        tctx.font=a.size+'px serif'; tctx.globalAlpha=.7; tctx.fillText(a.sym, a.x+wob, a.y); tctx.globalAlpha=1;
      } else {
        const w=a.size, h=a.size*0.7, x=a.x+wob, y=a.y;
        tctx.fillStyle=hexToRgba(col,.16); tctx.strokeStyle=hexToRgba(col,.3);
        roundRectTB(x,y,w,h,10); tctx.fill();
        tctx.beginPath(); tctx.moveTo(x+12,y+h); tctx.lineTo(x+8,y+h+10); tctx.lineTo(x+24,y+h); tctx.fill();
        tctx.fillStyle=hexToRgba(col,.5);
        for(let d=0;d<3;d++){ tctx.beginPath(); tctx.arc(x+w*0.28+d*w*0.22, y+h/2, 3, 0, Math.PI*2); tctx.fill(); }
      }
    });
  } else if(tbTheme==='images'){
    tbAgents.forEach(a=>{
      const b=0.06+0.16*(0.5+0.5*Math.sin(tbT*a.sp+a.ph));
      tctx.fillStyle=hexToRgba(col,b);
      tctx.fillRect(a.x+1,a.y+1,a.cell-2,a.cell-2);
    });
  } else if(tbTheme==='bias'){
    const base=H*0.82, maxh=H*0.5;
    tbAgents.forEach(a=>{
      const h=maxh*(0.25+0.7*(0.5+0.5*Math.sin(tbT*a.sp+a.ph)));
      tctx.fillStyle=hexToRgba(col,.2);
      tctx.fillRect(a.x, base-h, a.bw, h);
      tctx.strokeStyle=hexToRgba(col,.3); tctx.strokeRect(a.x, base-h, a.bw, h);
    });
    tctx.strokeStyle=hexToRgba(col,.25);
    tctx.beginPath(); tctx.moveTo(0,base); tctx.lineTo(W,base); tctx.stroke();
  }
}
function tbLabel(t,cx_,cy_,s){ tctx.fillStyle=hexToRgba('#ffffff',.95); tctx.font='700 '+Math.round(s*0.5)+'px sans-serif'; tctx.textAlign='center'; tctx.textBaseline='middle'; tctx.fillText(t,cx_,cy_); }
function roundRectTB(x,y,w,h,r){ tctx.beginPath(); tctx.moveTo(x+r,y); tctx.arcTo(x+w,y,x+w,y+h,r); tctx.arcTo(x+w,y+h,x,y+h,r); tctx.arcTo(x,y+h,x,y,r); tctx.arcTo(x,y,x+w,y,r); tctx.closePath(); }

// ---------- Audio: first-gesture start + UI toggles ----------
function volIcon(){
  const p=AudioFX.prefs;
  if(p.muted || p.volume<=0) return '\uD83D\uDD07';      // muted
  if(p.volume < 0.34) return '\uD83D\uDD08';             // low
  if(p.volume < 0.67) return '\uD83D\uDD09';             // medium
  return '\uD83D\uDD0A';                                 // high
}
function refreshSoundButtons(){
  const mb=document.getElementById('musicBtn');
  const sb=document.getElementById('sfxBtn');
  const vb=document.getElementById('volBtn');
  const mu=document.getElementById('muteBtn');
  const sl=document.getElementById('volSlider');
  const pc=document.getElementById('volPct');
  if(mb){ mb.textContent = AudioFX.prefs.music ? '\uD83C\uDFB5' : '\uD83C\uDFB5\u0336'; mb.classList.toggle('off', !AudioFX.prefs.music); }
  if(sb){ sb.textContent = AudioFX.prefs.sfx ? '\uD83D\uDD0A' : '\uD83D\uDD07'; sb.classList.toggle('off', !AudioFX.prefs.sfx); }
  if(vb){ vb.textContent = volIcon(); vb.classList.toggle('off', AudioFX.prefs.muted || AudioFX.prefs.volume<=0); }
  if(mu){ mu.textContent = volIcon(); }
  const pct = Math.round(AudioFX.prefs.volume*100);
  if(sl){ sl.value = AudioFX.prefs.muted ? 0 : pct; }
  if(pc){ pc.textContent = (AudioFX.prefs.muted ? 0 : pct) + '%'; }
}
function toggleVolPanel(){
  AudioFX.resume();
  document.getElementById('volPanel').classList.toggle('show');
  AudioFX.play('click');
}
function setVolumePct(v){
  AudioFX.resume();
  AudioFX.setVolume((+v)/100);
  refreshSoundButtons();
}
function toggleMute(){
  AudioFX.resume();
  const nowMuted = !AudioFX.prefs.muted;
  AudioFX.setMuted(nowMuted);
  // if unmuting from 0 volume, give a sensible default level
  if(!nowMuted && AudioFX.prefs.volume<=0) AudioFX.setVolume(0.9);
  refreshSoundButtons();
  if(!AudioFX.prefs.muted) AudioFX.play('click');
}
// close the volume popover when clicking elsewhere
document.addEventListener('pointerdown', (e)=>{
  const wrap=document.querySelector('.vol-wrap');
  const panel=document.getElementById('volPanel');
  if(panel && panel.classList.contains('show') && wrap && !wrap.contains(e.target)){
    panel.classList.remove('show');
  }
});
function toggleMusic(){
  AudioFX.resume();
  AudioFX.setMusic(!AudioFX.prefs.music);
  // (re)start the appropriate track if music was just turned on
  if(AudioFX.prefs.music){
    const onPlay=document.getElementById('screen-play').classList.contains('active');
    if(onPlay && state.questions[state.cursor]){
      const q=state.questions[state.cursor];
      AudioFX.playMusic(q.lessonId || lessonIdFromTitle(q.lessonTitle));
    } else AudioFX.playMusic('menu');
  }
  refreshSoundButtons();
}
function toggleSfx(){
  AudioFX.resume();
  AudioFX.setSfx(!AudioFX.prefs.sfx);
  refreshSoundButtons();
  if(AudioFX.prefs.sfx) AudioFX.play('click');
}
// Start the audio context + menu music on the very first user interaction
// (browsers block autoplay until a gesture happens).
function primeAudio(){
  AudioFX.resume();
  // only start menu music if we're not already in a game
  if(!document.getElementById('screen-play').classList.contains('active')){
    AudioFX.playMusic('menu');
  }
  window.removeEventListener('pointerdown', primeAudio);
  window.removeEventListener('keydown', primeAudio);
}
window.addEventListener('pointerdown', primeAudio);
window.addEventListener('keydown', primeAudio);

// init
renderPlayers();
refreshSoundButtons();
