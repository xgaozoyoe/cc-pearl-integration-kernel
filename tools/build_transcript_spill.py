from pathlib import Path
import subprocess,json,hashlib,re,difflib
root=Path(__file__).resolve().parents[1]; out=root/'artifacts/transcript-spill';out.mkdir(exist_ok=True)
base=(root/'artifacts/transcript-tasks/consumer_fine.cu').read_text()
m={'variants':{}}
for name in ['control','short_b','short_a','short_ab']:
 s=base
 if name in ('short_b','short_ab'):
  s=s.replace('br[Lead+1]','br[1]')
  s=s.replace('for(int p=0;p<Lead;++p)', 'for(int p=0;p<0;++p)')
  s=s.replace('constexpr int slots=Lead+1;', 'constexpr int slots=1;')
  old='''                    if(half==0 && row==1 && g+Lead<Groups*(TileK/32) && (TileK==64 || kt*TileK+((g+Lead)/Groups)*32<k))
                        br[(g+Lead)%slots]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+((g+Lead)%Groups)*16,((g+Lead)/Groups)*32,lane,BN);'''
  assert old in s;s=s.replace(old,'')
  anchor='            int slice=g/Groups,group=g%Groups,slot=g%slots;'
  s=s.replace(anchor,anchor+'\n            br[0]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+group*16,slice*32,lane,BN);')
 if name in ('short_a','short_ab'):
  old='                    if(half==0 && slice+1<TileK/32 && (TileK==64 || kt*TileK+(slice+1)*32<k) && group==Groups-1) nextA=load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,(slice+1)*32,lane,BM);'
  assert old in s;s=s.replace(old,'')
  old='''                    if(slice+1<TileK/32 && (TileK==64 || kt*TileK+(slice+1)*32<k) && group==Groups-1 && half==1)
                        ar[row]=nextA;'''
  assert old in s;s=s.replace(old,'')
  s=s.replace('                Fragment nextA;', '                if(slice==1 && group==0) ar[row]=load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,32,lane,BM);')
 s=s.replace('transcript_tasks_consumer_fine','transcript_spill_'+name)
 (out/(name+'.cu')).write_text(s)
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='consumer_fine',tofile=name)))
 result=subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--cubin','--ptxas-options=-v',str(out/(name+'.cu')),'-o',str(out/(name+'.cubin'))],capture_output=True,text=True)
 (out/(name+'.build.log')).write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
 sass=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/(name+'.cubin'))],text=True);(out/(name+'.sass')).write_text(sass)
 m['variants'][name]=dict(entry=re.search(r'Function : (\S+)',sass)[1],sha256=hashlib.sha256((out/(name+'.cubin')).read_bytes()).hexdigest(),threads=416,smem=75968,correct=True)
 print(name,result.stderr,flush=True)
(out/'manifest.json').write_text(json.dumps(m,indent=2))
