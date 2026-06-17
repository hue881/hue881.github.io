// ============================================================
//  BRAINCADE — Audio Engine (Web Audio API, no external files)
//  - Synthesized UI sound effects (click, correct, wrong, etc.)
//  - Looping chiptune music: one theme per topic + a menu theme
//  - Master mute + music/sfx toggles (in-memory)
// ============================================================

const AudioFX = (() => {
  let ctx = null;
  let master, musicGain, sfxGain;
  let started = false;

  // In-memory preferences (no localStorage — the preview iframe forbids it).
  const prefs = {
    muted: false,
    music: true,    // default on
    sfx:   true,    // default on
    volume: 0.9,    // master volume 0..1
  };

  // ---- lazy init (must follow a user gesture for autoplay policies) ----
  function ensure(){
    if(ctx) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    ctx = new AC();
    master = ctx.createGain();   master.gain.value = prefs.muted ? 0 : prefs.volume;
    musicGain = ctx.createGain(); musicGain.gain.value = prefs.music ? 0.32 : 0;
    sfxGain = ctx.createGain();   sfxGain.gain.value = prefs.sfx ? 0.9 : 0;
    musicGain.connect(master);
    sfxGain.connect(master);
    master.connect(ctx.destination);
  }
  function resume(){ ensure(); if(ctx.state === 'suspended') ctx.resume(); started = true; }

  // ---- generic one-shot tone for SFX ----
  function tone({freq=440, type='square', dur=0.12, gain=0.5, slideTo=null, attack=0.005, release=0.06, when=0}){
    if(!ctx) return;
    const t0 = ctx.currentTime + when;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, t0);
    if(slideTo) osc.frequency.exponentialRampToValueAtTime(Math.max(1,slideTo), t0 + dur);
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(gain, t0 + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur + release);
    osc.connect(g); g.connect(sfxGain);
    osc.start(t0); osc.stop(t0 + dur + release + 0.02);
  }

  // ---- SFX library ----
  const SFX = {
    click(){ tone({freq:520, type:'square', dur:0.04, gain:0.35, slideTo:660}); },
    select(){ tone({freq:440, type:'square', dur:0.06, gain:0.4, slideTo:880}); },
    nav(){ tone({freq:300, type:'triangle', dur:0.10, gain:0.4, slideTo:520}); },
    add(){ tone({freq:660, type:'square', dur:0.05, gain:0.4, slideTo:990}); tone({freq:990,type:'square',dur:0.06,gain:0.3,when:0.05}); },
    remove(){ tone({freq:400, type:'square', dur:0.08, gain:0.35, slideTo:180}); },
    correct(){ // happy arpeggio C-E-G-C
      const seq=[523.25,659.25,783.99,1046.5];
      seq.forEach((f,i)=>tone({freq:f,type:'square',dur:0.10,gain:0.4,when:i*0.075}));
    },
    wrong(){ tone({freq:240,type:'sawtooth',dur:0.22,gain:0.4,slideTo:90}); },
    timeup(){ tone({freq:330,type:'sawtooth',dur:0.16,gain:0.35,slideTo:160}); tone({freq:200,type:'sawtooth',dur:0.24,gain:0.35,slideTo:80,when:0.14}); },
    tick(){ tone({freq:1200,type:'square',dur:0.02,gain:0.18}); },
    whoosh(){ tone({freq:200,type:'triangle',dur:0.18,gain:0.3,slideTo:760}); },
    fanfare(){
      const seq=[523.25,523.25,523.25,659.25,783.99,1046.5];
      const times=[0,0.12,0.24,0.40,0.55,0.72];
      seq.forEach((f,i)=>{ tone({freq:f,type:'square',dur:0.16,gain:0.45,when:times[i]});
                           tone({freq:f/2,type:'triangle',dur:0.16,gain:0.25,when:times[i]}); });
    }
  };
  function play(name){ if(!ctx || !prefs.sfx) return; (SFX[name]||SFX.click)(); }

  // ============================================================
  //  MUSIC — looping chiptune sequences (lead + bass per topic)
  //  Notes are frequencies; 0 = rest. Each step is a 16th-ish beat.
  // ============================================================
  const N = { // note table (Hz)
    C3:130.81,D3:146.83,E3:164.81,F3:174.61,G3:196.00,A3:220.00,B3:246.94,
    C4:261.63,D4:293.66,E4:329.63,F4:349.23,G4:392.00,A4:440.00,B4:493.88,
    C5:523.25,D5:587.33,E5:659.25,F5:698.46,G5:783.99,A5:880.00, R:0
  };
  const _ = N.R;

  // Each theme: lead melody, bass line, tempo (sec per step), lead waveform.
  const THEMES = {
    menu: {
      tempo:0.16, type:'square',
      lead:[N.C4,N.E4,N.G4,N.C5, N.B4,N.G4,N.E4,N.G4, N.A4,N.C5,N.E5,N.C5, N.G4,N.E4,N.C4,_],
      bass:[N.C3,_,N.G3,_, N.C3,_,N.G3,_, N.A3,_,N.E3,_, N.G3,_,N.C3,_]
    },
    // Robot maze: marching, mechanical
    maze: {
      tempo:0.15, type:'square',
      lead:[N.E4,N.E4,N.G4,N.E4, N.A4,N.G4,N.E4,_, N.D4,N.E4,N.G4,N.A4, N.G4,N.E4,N.D4,_],
      bass:[N.A3,_,N.A3,_, N.E3,_,N.E3,_, N.D3,_,N.D3,_, N.E3,_,N.A3,_]
    },
    // Neural nets: bright, curious, thoughtful
    neuron: {
      tempo:0.17, type:'triangle',
      lead:[N.D4,N.F4,N.A4,N.D5, N.C5,N.A4,N.F4,N.A4, N.B4,N.G4,N.D4,N.G4, N.A4,N.F4,N.D4,_],
      bass:[N.D3,_,N.A3,_, N.F3,_,N.A3,_, N.G3,_,N.D3,_, N.A3,_,N.D3,_]
    },
    // Recommendation: upbeat, poppy, ice-cream-truck cheerful
    recommend: {
      tempo:0.14, type:'square',
      lead:[N.G4,N.A4,N.B4,N.D5, N.B4,N.G4,N.B4,N.D5, N.E5,N.D5,N.B4,N.G4, N.A4,N.B4,N.A4,_],
      bass:[N.G3,_,N.D3,_, N.G3,_,N.D3,_, N.E3,_,N.B3,_, N.D3,_,N.G3,_]
    },
    // Images: dreamy, pixel-shimmer
    images: {
      tempo:0.18, type:'triangle',
      lead:[N.C5,N.E5,N.G5,N.E5, N.A4,N.C5,N.E5,N.C5, N.F4,N.A4,N.C5,N.A4, N.G4,N.E4,N.C4,_],
      bass:[N.C3,_,N.G3,_, N.A3,_,N.E3,_, N.F3,_,N.C3,_, N.G3,_,N.C3,_]
    },
    // Chatbot: bouncy, conversational
    chatbot: {
      tempo:0.15, type:'square',
      lead:[N.A4,N.G4,N.A4,N.C5, N.B4,N.A4,N.G4,N.E4, N.G4,N.A4,N.B4,N.C5, N.A4,N.G4,N.E4,_],
      bass:[N.A3,_,N.E3,_, N.G3,_,N.D3,_, N.E3,_,N.A3,_, N.E3,_,N.A3,_]
    },
    // Bias & fairness: serious, contemplative, minor feel
    bias: {
      tempo:0.18, type:'triangle',
      lead:[N.A4,N.C5,N.B4,N.A4, N.G4,N.A4,N.E4,_, N.F4,N.G4,N.A4,N.G4, N.E4,N.D4,N.A3,_],
      bass:[N.A3,_,N.E3,_, N.F3,_,N.C3,_, N.D3,_,N.A3,_, N.E3,_,N.A3,_]
    }
  };

  let musicNodes = [];   // active scheduled-loop handles
  let currentTheme = null;
  let loopTimer = null;

  function stopMusic(){
    if(loopTimer){ clearTimeout(loopTimer); loopTimer=null; }
    musicNodes.forEach(n=>{ try{ n.stop(); }catch(e){} });
    musicNodes = [];
    currentTheme = null;
  }

  function scheduleBar(theme, startAt){
    const T = THEMES[theme]; if(!T) return 0;
    const step = T.tempo;
    const len = T.lead.length;
    for(let i=0;i<len;i++){
      const t = startAt + i*step;
      const lf = T.lead[i];
      if(lf>0){
        const o=ctx.createOscillator(), g=ctx.createGain();
        o.type=T.type; o.frequency.setValueAtTime(lf,t);
        g.gain.setValueAtTime(0.0001,t);
        g.gain.exponentialRampToValueAtTime(0.5,t+0.01);
        g.gain.exponentialRampToValueAtTime(0.0001,t+step*0.9);
        o.connect(g); g.connect(musicGain);
        o.start(t); o.stop(t+step);
        musicNodes.push(o);
      }
      const bf = T.bass[i];
      if(bf>0){
        const o=ctx.createOscillator(), g=ctx.createGain();
        o.type='triangle'; o.frequency.setValueAtTime(bf,t);
        g.gain.setValueAtTime(0.0001,t);
        g.gain.exponentialRampToValueAtTime(0.4,t+0.01);
        g.gain.exponentialRampToValueAtTime(0.0001,t+step*1.4);
        o.connect(g); g.connect(musicGain);
        o.start(t); o.stop(t+step*1.6);
        musicNodes.push(o);
      }
    }
    return len*step;
  }

  function playMusic(theme){
    ensure();
    if(currentTheme === theme) return;          // already playing this theme
    stopMusic();
    if(!THEMES[theme]) return;
    currentTheme = theme;
    // keep only a small window of scheduled nodes; reschedule each bar
    const loop = () => {
      if(currentTheme !== theme) return;
      // prune finished nodes occasionally
      if(musicNodes.length > 400) musicNodes = musicNodes.slice(-200);
      const barLen = scheduleBar(theme, ctx.currentTime + 0.05);
      loopTimer = setTimeout(loop, barLen*1000 - 60);
    };
    loop();
  }

  // ---- preference setters ----
  function setMuted(m){
    prefs.muted=m;
    if(master) master.gain.setTargetAtTime(m?0:prefs.volume, ctx.currentTime, 0.02);
  }
  function setVolume(v){
    prefs.volume = Math.max(0, Math.min(1, v));
    // setting volume above 0 implicitly unmutes
    if(prefs.volume>0) prefs.muted=false;
    else prefs.muted=true;
    if(master) master.gain.setTargetAtTime(prefs.muted?0:prefs.volume, ctx.currentTime, 0.02);
  }
  function setMusic(on){
    prefs.music=on;
    if(musicGain) musicGain.gain.setTargetAtTime(on?0.32:0, ctx.currentTime, 0.05);
  }
  function setSfx(on){
    prefs.sfx=on;
    if(sfxGain) sfxGain.gain.setTargetAtTime(on?0.9:0, ctx.currentTime, 0.05);
  }

  return {
    resume, play, playMusic, stopMusic,
    setMuted, setMusic, setSfx, setVolume,
    get prefs(){ return prefs; }
  };
})();
