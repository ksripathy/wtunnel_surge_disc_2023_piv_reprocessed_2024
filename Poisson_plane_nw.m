function[u]=Poisson_plane_nw(Mask_h,fx,fxx,fy,fyy,f0,h1,h2,Left,Right,Top,Bottom)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Comments by Daniele Ragni, June 2016 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % Standard orientation of the gradients due to dx0 pos and dy0 neg
% Mask_h=Mask;fx=dPdx;fxx=dP2dx2;fy=dPdr;fyy=dP2dr2;f0=PBern;h1=dx;h2=dr;Left='Dir';Right='Nx';Top='Dir';Bottom='Nx';  

tic
n1=size(Mask_h,2);
n2=size(Mask_h,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Identification of the points to be assigned as boundaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BL=[]; bl=[]; BR=[]; br=[]; BT=[]; bt=[]; BB=[]; bb=[];
count=0;
for r=1:n2
    Dum=find(Mask_h(r,:)==1);
    % Left/right
    if ~isempty(Dum)
        count=count+1;
        BL(count,1:3)=[(Dum(1)-1)*n2+r,r,Dum(1)];
        BR(count,1:3)=[(Dum(end)-1)*n2+r,r,Dum(end)];
        Dum2=Mask_h(r,Dum(1):Dum(end));
        % Internal points
        Diff2=diff(Dum2);
        R =find(Diff2==-1)';L=find(Diff2==+1)';
        if ~isempty(R); Loc=R+Dum(1)-1; br=[br; [(Loc-1)*n2+r,rectpulse(r,length(R)),Loc]]; end;
        if ~isempty(L); Loc=L+Dum(1);   bl=[bl; [(Loc-1)*n2+r,rectpulse(r,length(L)),Loc]]; end;
    end
end
% Points on the top/bottom boundary
count=0;
for c=1:n1
    Dum=find(Mask_h(:,c)==1);
    % Top/down
    if ~isempty(Dum)
        count=count+1;
        BT(count,1:3)=[(c-1)*n2+Dum(1),Dum(1),c];
        BB(count,1:3)=[(c-1)*n2+Dum(end),Dum(end),c];
        Dum2=Mask_h(Dum(1):Dum(end),c);
        % Internal points
        Diff2=diff(Dum2);
        T=find(Diff2==-1); B=find(Diff2==+1);
        if ~isempty(T); Loc=T+Dum(1)-1; bb=[bb; [(c-1)*n2+Loc,Loc,rectpulse(c,length(T))]]; end;
        if ~isempty(B); Loc=B+Dum(1);   bt=[bt; [(c-1)*n2+Loc,Loc,rectpulse(c,length(B))]]; end;      
    end
end
% Unique boundaries, priority to horizontal direction
BT=setdiff(BT,BL,'rows'); BT=setdiff(BT,BR,'rows');
BB=setdiff(BB,BL,'rows'); BB=setdiff(BB,BR,'rows');
bt=setdiff(bt,bl,'rows'); bt=setdiff(bt,br,'rows');
bb=setdiff(bb,bl,'rows'); bb=setdiff(bb,br,'rows');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Identification of the external boundary type. Follow Dani's scheme
% Boundary BT:
%--------------------------------------------------------------------------
All_in=[BT;BB;BL;BR]; if ~isempty(All_in); All_in(:,4)=0; end;
Dum=zeros([size(All_in,1),4]); %-n2  -1  +1  +n2
for count=1:size(All_in,1)
    % Rows and columns
    if All_in(count,3)~=1; Dum(count,1)=Mask_h(All_in(count,1)-n2); else Dum(count,1)=0; end;
    if All_in(count,3)~=n1; Dum(count,4)=Mask_h(All_in(count,1)+n2); else Dum(count,4)=0; end;
    if All_in(count,2)~=1; Dum(count,2)=Mask_h(All_in(count,1)-1); else Dum(count,2)=0; end;
    if All_in(count,2)~=n2; Dum(count,3)=Mask_h(All_in(count,1)+1); else Dum(count,3)=0; end;
    % Identification
    if isequal(Dum(count,:),[0,1,1,1]); All_in(count,4)=11; end;
    if isequal(Dum(count,:),[1,1,0,1]); All_in(count,4)=12; end;
    if isequal(Dum(count,:),[1,1,1,0]); All_in(count,4)=13; end;
    if isequal(Dum(count,:),[1,0,1,1]); All_in(count,4)=14; end;
    if isequal(Dum(count,:),[0,1,0,1]); All_in(count,4)=21; end;
    if isequal(Dum(count,:),[1,1,0,0]); All_in(count,4)=22; end;
    if isequal(Dum(count,:),[1,0,1,0]); All_in(count,4)=23; end;
    if isequal(Dum(count,:),[0,0,1,1]); All_in(count,4)=24; end;
    if isequal(Dum(count,:),[0,1,0,0]); All_in(count,4)=31; end;
    if isequal(Dum(count,:),[1,0,0,0]); All_in(count,4)=32; end;
    if isequal(Dum(count,:),[0,0,1,0]); All_in(count,4)=33; end;
    if isequal(Dum(count,:),[0,0,0,1]); All_in(count,4)=34; end;
end
All_in=unique(All_in,'rows');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Identification of the internal boundary type. Follow Dani's scheme
% Boundary BT:
%--------------------------------------------------------------------------
All_inI=[bb;bt;bl;br]; if ~isempty(All_inI); All_inI(:,4)=0; end;
DumI=zeros([size(All_inI,1),4]); %-n2  -1  +1  +n2
for count=1:size(All_inI,1)
    % Righe
    if All_inI(count,3)~=1; DumI(count,1)=Mask_h(All_inI(count,1)-n2); else DumI(count,1)=0; end;
    if All_inI(count,3)~=n1; DumI(count,4)=Mask_h(All_inI(count,1)+n2); else DumI(count,4)=0; end;
    if All_inI(count,2)~=1; DumI(count,2)=Mask_h(All_inI(count,1)-1); else DumI(count,2)=0; end;
    if All_inI(count,2)~=n2; DumI(count,3)=Mask_h(All_inI(count,1)+1); else DumI(count,3)=0; end;
    % Identification
    if isequal(DumI(count,:),[0,1,1,1]); All_inI(count,4)=11; end;
    if isequal(DumI(count,:),[1,1,0,1]); All_inI(count,4)=12; end;
    if isequal(DumI(count,:),[1,1,1,0]); All_inI(count,4)=13; end;
    if isequal(DumI(count,:),[1,0,1,1]); All_inI(count,4)=14; end;
    if isequal(DumI(count,:),[0,1,0,1]); All_inI(count,4)=21; end;
    if isequal(DumI(count,:),[1,1,0,0]); All_inI(count,4)=22; end;
    if isequal(DumI(count,:),[1,0,1,0]); All_inI(count,4)=23; end;
    if isequal(DumI(count,:),[0,0,1,1]); All_inI(count,4)=24; end;
    if isequal(DumI(count,:),[0,1,0,0]); All_inI(count,4)=31; end;
    if isequal(DumI(count,:),[1,0,0,0]); All_inI(count,4)=32; end;
    if isequal(DumI(count,:),[0,0,1,0]); All_inI(count,4)=33; end;
    if isequal(DumI(count,:),[0,0,0,1]); All_inI(count,4)=34; end;
end
All_inI=unique(All_inI,'rows');
%--------------------------------------------------------------------------
% Removing from the list the points with Dirichlet boundaries
index_Dir=[];
if strcmp(Left,'Dir');
    index_Dir=[index_Dir; All_in((All_in(:,4)==11)|(All_in(:,4)==34)|(All_in(:,4)==24)|(All_in(:,4)==21),1)];
    All_in((All_in(:,4)==11)|(All_in(:,4)==34)|(All_in(:,4)==24)|(All_in(:,4)==21),4)=10;
end;
if strcmp(Right,'Dir');
    index_Dir=[index_Dir; All_in((All_in(:,4)==13)|(All_in(:,4)==32)|(All_in(:,4)==22)|(All_in(:,4)==23),1)];
    All_in((All_in(:,4)==13)|(All_in(:,4)==32)|(All_in(:,4)==22)|(All_in(:,4)==23),4)=10;
end
if strcmp(Top,'Dir');
    index_Dir=[index_Dir; All_in((All_in(:,4)==14)|(All_in(:,4)==33)|(All_in(:,4)==23)|(All_in(:,4)==24),1)];
    All_in((All_in(:,4)==14)|(All_in(:,4)==33)|(All_in(:,4)==23)|(All_in(:,4)==24),4)=10;
end
if strcmp(Bottom,'Dir');
    index_Dir=[index_Dir; All_in((All_in(:,4)==12)|(All_in(:,4)==31)|(All_in(:,4)==21)|(All_in(:,4)==22),1)];
    All_in((All_in(:,4)==12)|(All_in(:,4)==31)|(All_in(:,4)==21)|(All_in(:,4)==22),4)=10;
end
index_Dir=unique(index_Dir,'rows'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot Updated Mask
Mask=Mask_h; 
% External conditions
if ~isempty(All_in(:,4)==10);        Mask(All_in(All_in(:,4)==10,1))=10;  end;
if ~isempty(All_in(:,4)==13);        Mask(All_in(All_in(:,4)==13,1))=-4;       end;
if ~isempty(All_in(:,4)==11);        Mask(All_in(All_in(:,4)==11,1))=+4;       end;
if ~isempty(All_in(:,4)==12);        Mask(All_in(All_in(:,4)==12,1))=-6;       end;
if ~isempty(All_in(:,4)==14);        Mask(All_in(All_in(:,4)==14,1))=+6;       end;
% Corners
if ~isempty(All_in(:,4)==24);        Mask(All_in(All_in(:,4)==24,1))=2.4;       end;
if ~isempty(All_in(:,4)==23);        Mask(All_in(All_in(:,4)==23,1))=2.3;       end;
if ~isempty(All_in(:,4)==21);        Mask(All_in(All_in(:,4)==21,1))=2.1;       end;
if ~isempty(All_in(:,4)==22);        Mask(All_in(All_in(:,4)==22,1))=2.2;       end;
% Corners
if ~isempty(All_in(:,4)==31);        Mask(All_in(All_in(:,4)==31,1))=3.1;       end;
if ~isempty(All_in(:,4)==32);        Mask(All_in(All_in(:,4)==32,1))=3.2;       end;
if ~isempty(All_in(:,4)==33);        Mask(All_in(All_in(:,4)==33,1))=3.3;       end;
if ~isempty(All_in(:,4)==34);        Mask(All_in(All_in(:,4)==34,1))=3.4;       end;
% Internal conditions
if ~isempty(All_inI)
    if ~isempty(All_inI(:,4)==13);        Mask(All_inI(All_inI(:,4)==13,1))=-2;       end;
    if ~isempty(All_inI(:,4)==11);        Mask(All_inI(All_inI(:,4)==11,1))=+2;       end;
    if ~isempty(All_inI(:,4)==12);        Mask(All_inI(All_inI(:,4)==12,1))=-3;       end;
    if ~isempty(All_inI(:,4)==14);        Mask(All_inI(All_inI(:,4)==14,1))=+3;       end;
    % Corners
    if ~isempty(All_inI(:,4)==24);        Mask(All_inI(All_inI(:,4)==24,1))=2.4;       end;
    if ~isempty(All_inI(:,4)==23);        Mask(All_inI(All_inI(:,4)==23,1))=2.3;       end;
    if ~isempty(All_inI(:,4)==21);        Mask(All_inI(All_inI(:,4)==21,1))=2.1;       end;
    if ~isempty(All_inI(:,4)==22);        Mask(All_inI(All_inI(:,4)==22,1))=2.2;       end;
    % Corners
    if ~isempty(All_inI(:,4)==31);        Mask(All_inI(All_inI(:,4)==31,1))=3.1;       end;
    if ~isempty(All_inI(:,4)==32);        Mask(All_inI(All_inI(:,4)==32,1))=3.2;       end;
    if ~isempty(All_inI(:,4)==33);        Mask(All_inI(All_inI(:,4)==33,1))=3.3;       end;
    if ~isempty(All_inI(:,4)==34);        Mask(All_inI(All_inI(:,4)==34,1))=3.4;       end;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Starting ordering indices for assignment. The mask is created, from here
% we start the Poisson solver
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Finding Mask points, Dirichlet, external derivatives, corner points and
% internal boundaries
index_0=find(Mask_h==0);
index_D=index_Dir;
% External
index_dxp=All_in([find(All_in(:,4)==11); find(All_in(:,4)==34)],1);
index_dxm=All_in([find(All_in(:,4)==13); find(All_in(:,4)==32)],1);
index_dyp=All_in([find(All_in(:,4)==14); find(All_in(:,4)==33)],1);
index_dym=All_in([find(All_in(:,4)==12); find(All_in(:,4)==31)],1);
% External corners
index_TLC=All_in(All_in(:,4)==24,1); index_TRC=All_in(All_in(:,4)==23,1); index_BLC=All_in(All_in(:,4)==21,1); index_BRC=All_in(All_in(:,4)==22,1);
% Internal boundaries
if ~isempty(All_inI)
    if ~isempty([find(All_inI(:,4)==13); find(All_inI(:,4)==32)]); index_m2=All_inI([find(All_inI(:,4)==13); find(All_inI(:,4)==32)],1); else index_m2=[]; end;
    if ~isempty([find(All_inI(:,4)==11); find(All_inI(:,4)==34)]); index_p2=All_inI([find(All_inI(:,4)==11); find(All_inI(:,4)==34)],1); else index_p2=[]; end;
    if ~isempty([find(All_inI(:,4)==12); find(All_inI(:,4)==31)]); index_m3=All_inI([find(All_inI(:,4)==12); find(All_inI(:,4)==31)],1); else index_m3=[]; end;
    if ~isempty([find(All_inI(:,4)==14); find(All_inI(:,4)==33)]); index_p3=All_inI([find(All_inI(:,4)==14); find(All_inI(:,4)==33)],1); else index_p3=[]; end;
    % Internal corners
    if ~isempty(All_inI(:,4)==24); index_tlc=All_inI(All_inI(:,4)==24,1); else index_tlc=[]; end;
    if ~isempty(All_inI(:,4)==23); index_trc=All_inI(All_inI(:,4)==23,1); else index_trc=[]; end;
    if ~isempty(All_inI(:,4)==21); index_blc=All_inI(All_inI(:,4)==21,1); else index_blc=[]; end;
    if ~isempty(All_inI(:,4)==22); index_brc=All_inI(All_inI(:,4)==22,1); else index_brc=[]; end;
else
    index_tlc=[]; index_blc=[];index_brc=[]; index_trc=[]; index_m2=[]; index_m3=[]; index_p2=[]; index_p3=[];
end  
% Indices of the points to be recontructed points to reconstruct
Mat_dif=[];
if ~isempty(index_0); Mat_dif=[Mat_dif; index_0]; end;
if ~isempty(All_in);  Mat_dif=[Mat_dif; All_in(:,1)]; end;
if ~isempty(All_inI); Mat_dif=[Mat_dif; All_inI(:,1)]; end;
index_P=setdiff(1:n1*n2,Mat_dif)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Indices arrangement
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
index_PTLC=[]; index_PBLC=[]; index_PTRC=[]; index_PBRC=[];
index_Pdxp=[]; index_Pdxm=[]; index_Pdyp=[]; index_Pdym=[];
index_ptlc=[]; index_pblc=[]; index_ptrc=[]; index_pbrc=[];
index_ip2=[]; index_im2=[]; index_ip3=[]; index_im3=[]; 
% Mask points
if ~isempty(index_0); index_i0=[index_0,index_0,ones(length(index_0),1)]; else index_i0=[]; end;
% Dirichlet points
if ~isempty(index_D); index_iD=[index_D,index_D,ones(length(index_D),1)]; else index_iD=[]; end;

% Interior points
P=rectpulse(index_P,5); index_P=[P, P+repmat([-n2;-1;0;1;n2],length(index_P),1),repmat([1/h1^2; 1/h2^2; -2*(1/h2^2+1/h1^2); 1/h2^2; 1/h1^2],length(index_P),1)];
% Ghost points conditions of external corners on the internal points 
if ~isempty(index_TLC); P=rectpulse(index_TLC,3); index_PTLC=[P, P+repmat([0;+1;+n2],length(index_TLC),1),repmat([-2*(1/h2^2+1/h1^2); 1/h2^2; 1/h1^2],length(index_TLC),1)]; end;
if ~isempty(index_BLC); P=rectpulse(index_BLC,3); index_PBLC=[P, P+repmat([-1;0;+n2],length(index_BLC),1),repmat([1/h2^2; -2*(1/h2^2+1/h1^2); 1/h1^2],length(index_BLC),1)]; end;
if ~isempty(index_TRC); P=rectpulse(index_TRC,3); index_PTRC=[P, P+repmat([-n2;0;+1],length(index_TRC),1),repmat([1/h1^2; -2*(1/h2^2+1/h1^2); 1/h2^2],length(index_TRC),1)]; end;
if ~isempty(index_BRC); P=rectpulse(index_BRC,3); index_PBRC=[P, P+repmat([-n2;-1;0],length(index_BRC),1),repmat([1/h1^2; 1/h2^2; -2*(1/h2^2+1/h1^2)],length(index_BRC),1)]; end;
% Ghost points conditions of external boundaries on the internal points
if ~isempty(index_dxp); P=rectpulse(index_dxp,2); index_Pdxp=[P, P+repmat([0;+n2],length(index_dxp),1),repmat([-1; 1],length(index_dxp),1)]; end;
if ~isempty(index_dxm); P=rectpulse(index_dxm,2); index_Pdxm=[P, P+repmat([-n2;0],length(index_dxm),1),repmat([1; -1],length(index_dxm),1)]; end;
if ~isempty(index_dyp); P=rectpulse(index_dyp,2); index_Pdyp=[P, P+repmat([0;+1],length(index_dyp),1),repmat([-1; 1],length(index_dyp),1)]; end;
if ~isempty(index_dym); P=rectpulse(index_dym,2); index_Pdym=[P, P+repmat([-1;0],length(index_dym),1),repmat([1; -1],length(index_dym),1)]; end;
% Ghost points conditions of internal boundaries on the internal points
if ~isempty(index_p2); P=rectpulse(index_p2,2); index_ip2=[P, P+repmat([0;n2],length(index_p2),1), repmat([-1;1],length(index_p2),1)]; end;
if ~isempty(index_m2); P=rectpulse(index_m2,2); index_im2=[P, P+repmat([-n2;0],length(index_m2),1), repmat([1;-1],length(index_m2),1)];end;
if ~isempty(index_p3); P=rectpulse(index_p3,2); index_ip3=[P, P+repmat([0;1],length(index_p3),1), repmat([-1;1],length(index_p3),1)];end;
if ~isempty(index_m3); P=rectpulse(index_m3,2); index_im3=[P, P+repmat([-1;0],length(index_m3),1), repmat([1;-1],length(index_m3),1)]; end;
% Ghost points conditions of internal boundaries on the internal points
if ~isempty(index_tlc); P=rectpulse(index_tlc,3); index_ptlc=[P, P+repmat([0;+1;+n2],length(index_tlc),1),repmat([-2*(1/h2^2+1/h1^2); 1/h2^2; 1/h1^2],length(index_tlc),1)]; end;
if ~isempty(index_blc); P=rectpulse(index_blc,3); index_pblc=[P, P+repmat([-1;0;+n2],length(index_blc),1),repmat([1/h2^2; -2*(1/h2^2+1/h1^2); 1/h1^2],length(index_blc),1)]; end;
if ~isempty(index_trc); P=rectpulse(index_trc,3); index_ptrc=[P, P+repmat([-n2;0;+1],length(index_trc),1),repmat([1/h1^2; -2*(1/h2^2+1/h1^2); 1/h2^2],length(index_trc),1)]; end;
if ~isempty(index_brc); P=rectpulse(index_brc,3); index_pbrc=[P, P+repmat([-n2;-1;0],length(index_brc),1),repmat([1/h1^2; 1/h2^2; -2*(1/h2^2+1/h1^2)],length(index_brc),1)]; end;

% All conditions on the internal points
index_iP=[index_P; index_PTLC; index_PBLC; index_PTRC; index_PBRC; ...
          index_Pdxp; index_Pdxm; index_Pdyp; index_Pdym; ...
          index_ip2; index_im2; index_ip3; index_im3; ...
          index_ptlc; index_pblc; index_ptrc; index_pbrc];
      
% Creating the ghost points and assigning their conditions
index_tlce=[]; index_blce=[]; index_trce=[]; index_brce=[];
index_tlcm=[]; index_blcm=[]; index_trcm=[]; index_brcm=[];


% Ghosts external corners
count=0;
if  ~isempty(index_TLC);    
    kk=(1:length(index_TLC))';  Loc=n1*n2+count+(kk-1)*2;
    Conds=rectpulse([+1/h1^2; +1/(2*h1); -1/(2*h1); +1/h2^2; +1/(2*h2); -1/(2*h2)],length(index_TLC));
    Dum=[index_TLC(kk),Loc+1;    Loc+1,index_TLC(kk)+n2;     Loc+1,Loc+1;
         index_TLC(kk),Loc+2;    Loc+2,index_TLC(kk)+1;      Loc+2,Loc+2];
    index_tlce=[Dum,Conds];
    count=count+length(index_TLC)*2;
end

if  ~isempty(index_BLC);
    kk=(1:length(index_BLC))'; Loc=n1*n2+count+(kk-1)*2;
    Conds=rectpulse([+1/h1^2; +1/(2*h1); -1/(2*h1); +1/h2^2; +1/(2*h2); -1/(2*h2)],length(index_BLC));
    Dum=[index_BLC(kk),Loc+1;    Loc+1,index_BLC(kk)+n2;     Loc+1,Loc+1;
         index_BLC(kk),Loc+2;    Loc+2,index_BLC(kk)-1;      Loc+2,Loc+2];
    index_blce=[Dum,Conds];
    count=count+length(index_BLC)*2;
end

if  ~isempty(index_TRC);
    kk=(1:length(index_TRC))'; Loc=n1*n2+count+(kk-1)*2;
    Conds=rectpulse([+1/h1^2; +1/(2*h1); -1/(2*h1); +1/h2^2; +1/(2*h2); -1/(2*h2)],length(index_TRC));
    Dum=[index_TRC(kk),Loc+1;    Loc+1,index_TRC(kk)-n2;     Loc+1,Loc+1;
         index_TRC(kk),Loc+2;    Loc+2,index_TRC(kk)+1;      Loc+2,Loc+2];
    index_trce=[Dum,Conds];
    count=count+length(index_TRC)*2;
end
    
if  ~isempty(index_BRC);
    kk=(1:length(index_BRC))'; Loc=n1*n2+count+(kk-1)*2;
    Conds=rectpulse([+1/h1^2; +1/(2*h1); -1/(2*h1); +1/h2^2; +1/(2*h2); -1/(2*h2)],length(index_BRC));
    Dum=[index_BRC(kk),Loc+1;    Loc+1,index_BRC(kk)-n2;     Loc+1,Loc+1;
         index_BRC(kk),Loc+2;    Loc+2,index_BRC(kk)-1;      Loc+2,Loc+2];
    index_brce=[Dum,Conds];
    count=count+length(index_BRC)*2;
end

% Ghosts internal corners
if  ~isempty(index_tlc);    
    kk=(1:length(index_tlc))';  Loc=n1*n2+count+(kk-1)*2;
    Conds=rectpulse([+1/h1^2; +1/(2*h1); -1/(2*h1); +1/h2^2; +1/(2*h2); -1/(2*h2)],length(index_tlc));
    Dum=[index_tlc(kk),Loc+1;    Loc+1,index_tlc(kk)+n2;     Loc+1,Loc+1;
         index_tlc(kk),Loc+2;    Loc+2,index_tlc(kk)+1;      Loc+2,Loc+2];
    index_tlcm=[Dum,Conds];
    count=count+length(index_tlc)*2;
end
  
if  ~isempty(index_blc);    
    kk=(1:length(index_blc))';  Loc=n1*n2+count+(kk-1)*2;
    Conds=rectpulse([+1/h1^2; +1/(2*h1); -1/(2*h1); +1/h2^2; +1/(2*h2); -1/(2*h2)],length(index_blc));
    Dum=[index_blc(kk),Loc+1;    Loc+1,index_blc(kk)+n2;     Loc+1,Loc+1;
         index_blc(kk),Loc+2;    Loc+2,index_blc(kk)-1;      Loc+2,Loc+2];
    index_blcm=[Dum,Conds];
    count=count+length(index_blc)*2;
end

if  ~isempty(index_trc);    
    kk=(1:length(index_trc))';  Loc=n1*n2+count+(kk-1)*2;
    Conds=rectpulse([+1/h1^2; +1/(2*h1); -1/(2*h1); +1/h2^2; +1/(2*h2); -1/(2*h2)],length(index_trc));
    Dum=[index_trc(kk),Loc+1;    Loc+1,index_trc(kk)-n2;     Loc+1,Loc+1;
         index_trc(kk),Loc+2;    Loc+2,index_trc(kk)+1;      Loc+2,Loc+2];
    index_trcm=[Dum,Conds];
    count=count+length(index_trc)*2;
end

if  ~isempty(index_brc);    
    kk=(1:length(index_brc))';  Loc=n1*n2+count+(kk-1)*2;
    Conds=rectpulse([+1/h1^2; +1/(2*h1); -1/(2*h1); +1/h2^2; +1/(2*h2); -1/(2*h2)],length(index_brc));
    Dum=[index_brc(kk),Loc+1;    Loc+1,index_brc(kk)-n2;     Loc+1,Loc+1;
         index_brc(kk),Loc+2;    Loc+2,index_brc(kk)-1;      Loc+2,Loc+2];
    index_brcm=[Dum,Conds];
    count=count+length(index_brc)*2;
end

% Ghost points mask to allow placing both derivatives
index_g=[index_tlce;index_blce;index_trce;index_brce;...
         index_tlcm;index_blcm;index_trcm;index_brcm];

% Summarizing all points and building the matrix
Indices=[index_i0; index_iD; index_iP; index_g];
dim=(n1*n2)+count;
L=sparse(Indices(:,1),Indices(:,2),Indices(:,3),dim,dim);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forcing functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Applying the forcing functions
f2=fxx+fyy;
ub=zeros(dim,1); ub(1:n1*n2)=f2(:);
% Mask points
ub(index_0)=0;
ub(index_dxp)=h1^2/2*fxx(index_dxp)+h1*fx(index_dxp);
ub(index_dxm)=h1^2/2*fxx(index_dxm)-h1*fx(index_dxm);
ub(index_dyp)=h2^2/2*fyy(index_dyp)+h2*fy(index_dyp);
ub(index_dym)=h2^2/2*fyy(index_dym)-h2*fy(index_dym);
ub(index_p2)=h1^2/2*fxx(index_p2)+h1*fx(index_p2);
ub(index_m2)=h1^2/2*fxx(index_m2)-h1*fx(index_m2);
ub(index_p3)=h2^2/2*fyy(index_p3)+h2*fy(index_p3);
ub(index_m3)=h2^2/2*fyy(index_m3)-h2*fy(index_m3);

% Dirichlet points
ub(index_D)=f0(index_D);
% Ghost points
if count~=0
    Dum=[];
    if ~isempty(index_TLC); Dum=[Dum; reshape([+fx(index_TLC)'; +fy(index_TLC)'],[2*length(index_TLC),1])]; ub(index_TLC)=f2(index_TLC); end;
    if ~isempty(index_BLC); Dum=[Dum; reshape([+fx(index_BLC)'; -fy(index_BLC)'],[2*length(index_BLC),1])]; ub(index_BLC)=f2(index_BLC); end;
    if ~isempty(index_TRC); Dum=[Dum; reshape([-fx(index_TRC)'; +fy(index_TRC)'],[2*length(index_TRC),1])]; ub(index_TRC)=f2(index_TRC); end;
    if ~isempty(index_BRC); Dum=[Dum; reshape([-fx(index_BRC)'; -fy(index_BRC)'],[2*length(index_BRC),1])]; ub(index_BRC)=f2(index_BRC); end;
    if ~isempty(index_tlc); Dum=[Dum; reshape([+fx(index_tlc)'; +fy(index_tlc)'],[2*length(index_tlc),1])]; ub(index_tlc)=f2(index_tlc); end;
    if ~isempty(index_blc); Dum=[Dum; reshape([+fx(index_blc)'; -fy(index_blc)'],[2*length(index_blc),1])]; ub(index_blc)=f2(index_blc); end;
    if ~isempty(index_trc); Dum=[Dum; reshape([-fx(index_trc)'; +fy(index_trc)'],[2*length(index_trc),1])]; ub(index_trc)=f2(index_trc); end;
    if ~isempty(index_brc); Dum=[Dum; reshape([-fx(index_brc)'; -fy(index_brc)'],[2*length(index_brc),1])]; ub(index_brc)=f2(index_brc); end;
    ub(n1*n2+1:end)=Dum;
end
u = L\ub;
u = reshape(u(1:n1*n2),n2,n1);
% 
% % Analysis
% AA=mean(u(u(:)~=0));
% [ux,uy]=gradient(u,h1,h2);
% [f0x,f0y]=gradient(f0.*Mask_final,h1,h2);
% % 
% close all;
% figure(1)
% subplot(2,4,1); 
% imagesc(u)
% if abs(AA)>1e7;  uu=(u-AA).*Mask_final; imagesc(uu); caxis([min(uu(uu(:)~=0)),max(uu(uu(:)~=0))]);
% else  caxis([0,max(f0(f0(:)~=0))]);
% end
% axis image;
% subplot(2,4,2); imagesc(Mask); axis image; 
% subplot(2,4,3); imagesc(ux.*Mask_final); ff=caxis; axis image;
% subplot(2,4,4); imagesc(uy.*Mask_final); caxis([-1,1]); axis image;
% 
% subplot(2,4,5); imagesc(f0.*Mask_final); caxis([0,max(f0(f0(:)~=0))]); axis image;
% subplot(2,4,6); imagesc(Mask_final); axis image;
% subplot(2,4,7); imagesc(Mask_final.*f0x); caxis(ff); axis image;
% subplot(2,4,8); imagesc(Mask_final.*f0y); caxis([-1,1]); axis image;
% 

 toc