 
n=10;
A = rherm(n);
%hermitian matrix
 if((A) == (A'))
    fprintf('both are equal \n')
 end
fid = fopen('data.txt', 'w+');
for i=0:size(A, 1)
    if(i==0)
       fprintf(fid, '%d ', size(A));
       fprintf(fid, '\n'); 
    else
    fprintf(fid, '%f ', A(i,:));
    fprintf(fid, '\n');
    end
end
fclose(fid);