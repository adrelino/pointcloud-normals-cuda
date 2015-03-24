function [diffMean] = err(test,gold,name,varargin)
if( size(test) ~= size(gold))
    error('argument error, test and gold have different dimensions');
end

if(nargin>3 && varargin{1}) %error of absolute values (e.g. for eigenvectors)
    Diff=abs(abs(test)-abs(gold));
else
    Diff=abs(test-gold);
end
diff=sum(Diff(:));
diffMean=mean(Diff(:));

if nargout==0
fprintf(['Numerical error: sum=%0.5e,  mean=%0.5e  testObj=%s\n'],diff,diffMean,name);
end

end

